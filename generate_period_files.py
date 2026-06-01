# -*- coding: utf-8 -*-
"""独立 period 预运算脚本。

这个脚本故意不依赖 FactorForge 项目内的任何模块，复制到任意目录后，只要本机
有 numpy/pandas，且能访问脚本顶部配置的 eod/dates.npy，就可以直接生成：

- period.npy：周期名称，例如 5_0、20_19、W_4。
- period_dates.npy：period x dates 的 True/False 换仓日矩阵。
- dates.npy：与 eod/dates.npy 对齐的日期标签。

路径和周期集合都在脚本顶部集中配置；命令行参数只用于临时覆盖。
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# 独立配置区
# =============================================================================

# 输入：只需要 EOD 目录下的 dates.npy。
DEFAULT_EOD_PATH = Path("D:/凯纳/原始数据/eod")

# 输出：三个 period 相关 npy 文件统一写到这里。
DEFAULT_PERIOD_PATH = Path("D:/凯纳/原始数据/period")

# 原脚本会把交易天数少于 min_day 的短周期并入下一期，这里保持原口径。
# 1 日换仓不参与这个过滤，1_0 应该是首日 False、之后每天 True。
DEFAULT_MIN_DAY = 2

# 默认预计算周期。20/21 已包含所有 offset。
DEFAULT_PERIOD_DICT: dict[Any, list[Any]] = {
    1: [0],
    2: list(range(2)),
    3: list(range(3)),
    4: list(range(4)),
    5: list(range(5)),
    10: list(range(10)),
    20: list(range(20)),
    21: list(range(21)),
    "W": [0, 1, 2, 3, 4],
    "2W": [0, 1, "0D", "1D", "2D", "3D", "4D", "7D", "8D", "9D", "10D", "11D"],
    "3W": [0, 1, 2],
    "4W": [0, 1, 2, 3],
    "5W": [0, 1, 2, 3, 4],
    "6W": [0, 1, 2, 3, 4, 5],
    "M": [0, -5],
    "W53": [0],
}


def decode_array(values: Iterable) -> list[str]:
    """把 npy 中可能存在的 bytes/string/int 混合日期统一解码为字符串。"""
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        out.append(str(value).strip())
    return out


def normalize_date_label(value: Any) -> str:
    """把日期标签统一成 YYYYMMDD，兼容 YYYY-MM-DD、YYYY/MM/DD 和 bytes。"""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip().replace("-", "").replace("/", "")
    return text if len(text) == 8 and text.isdigit() else str(value).strip()


class TradeCalendar(object):
    """离线交易日历。

    这段逻辑：先用自然日生成日历，
    再按周末、公历节日、农历节日和清明节气估算休市日。period 文件最终仍会
    裁剪回 eod/dates.npy，所以这个日历主要用于样本尾部补未来交易日。
    """

    today = None
    last_trade_date = None
    next_trade_date = None
    market_open = True
    calendar = None
    _now_date = datetime.date.today()

    def __init__(self, day_win=60):
        self.calendar = self.create_calendar_offline(day_win=day_win * 1.5)
        self.calendar["from_now"] = self.calendar["trade_date"].apply(lambda x: (x - self._now_date).days)
        self.calendar = self.calendar[self.calendar["from_now"] < day_win]
        if self.calendar["from_now"].max() < day_win * 0.95:
            calendar_offline = self.create_calendar_offline(day_win=day_win * 1.5)
            calendar_offline = calendar_offline[calendar_offline["trade_date"] > self.calendar["trade_date"].iloc[-1]]
            self.calendar = pd.concat([self.calendar, calendar_offline], ignore_index=True).sort_values(
                by="trade_date"
            )
            self.calendar["from_now"] = self.calendar["trade_date"].apply(lambda x: (x - self._now_date).days)

        self.calendar["trade_date"] = pd.to_datetime(self.calendar["trade_date"])
        con = self.calendar["from_now"] < 0
        self.last_trade_date = self.calendar[con]["trade_date"].iloc[-1]

        con = self.calendar["from_now"] > 0
        self.next_trade_date = self.calendar[con]["trade_date"].iloc[0]

        if 0 in self.calendar["from_now"].values:
            self.today = self.calendar[self.calendar["from_now"] == 0]["trade_date"].iloc[0]
        else:
            self.today = self.next_trade_date
            self.market_open = False

        self.calendar["month"] = self.calendar["trade_date"].apply(lambda x: x.month)
        self.calendar["week"] = self.calendar["trade_date"].apply(lambda x: x.weekofyear)
        for key in ["month", "week"]:
            self.calendar.loc[self.calendar[key] == self.calendar[key].shift(), key] = 0
            self.calendar.loc[self.calendar[key] != 0, key] = 1
            self.calendar[key] = self.calendar[key].cumsum()
        self.calendar["week_of"] = self.calendar.groupby("week")["week"].rank(method="first")
        self.calendar["month_of"] = self.calendar.groupby("month")["month"].rank(method="first")

    @staticmethod
    def create_calendar_offline(day_win=84):
        now_date = datetime.date.today()
        start = now_date - datetime.timedelta(days=day_win)
        end = now_date + datetime.timedelta(days=day_win)
        calendar_offline = pd.DataFrame(pd.date_range(start, end, freq="D"))
        calendar_offline.rename(columns={0: "trade_date"}, inplace=True)

        calendar_offline["week"] = calendar_offline["trade_date"].dt.dayofweek + 1
        calendar_offline["date_info"] = calendar_offline["trade_date"].apply(
            lambda x: "公历%s月%s日" % (x.month, x.day) + " 农历" + Lunar(x).ln_date_str() + Lunar(x).ln_jie()
        )

        holidays = {
            "公历1月1日": 3,
            "公历5月1日": 5,
            "公历10月1日": 7,
            "正月初一": 6,
            "五月初五": 3,
            "八月十五": 3,
            "清明": 3,
        }
        for holiday in holidays.keys():
            calendar_offline.loc[calendar_offline["date_info"].str.contains(holiday), "rest"] = holidays[holiday]

        calendar_offline.loc[calendar_offline["week"] > 5, "holiday"] = 1
        for i in calendar_offline.index[10:-10]:
            if calendar_offline.at[i, "rest"] > 0:
                calendar_offline.at[i, "holiday"] = 1
                week = calendar_offline.at[i, "week"]
                rest = calendar_offline.at[i, "rest"]
                if rest == 3:
                    if week <= 2:
                        calendar_offline.at[i - 1, "holiday"] = 1
                        calendar_offline.at[i - 2, "holiday"] = 1
                    elif (week > 3) & (week < 7):
                        calendar_offline.at[i + 1, "holiday"] = 1
                        calendar_offline.at[i + 2, "holiday"] = 1
                    else:
                        calendar_offline.at[i + 1, "holiday"] = 1
                else:
                    for j in range(1, int(rest)):
                        calendar_offline.at[i + j, "holiday"] = 1
                    if rest == 6:
                        calendar_offline.at[i - 1, "holiday"] = 1

        calendar_offline = calendar_offline[calendar_offline["holiday"] != 1][["trade_date"]]
        calendar_offline["trade_date"] = calendar_offline["trade_date"].apply(
            lambda x: datetime.date(year=x.year, month=x.month, day=x.day)
        )
        return calendar_offline

    def cal_day_delta(self, appoint_day, delta):
        try:
            if isinstance(appoint_day, str):
                appoint_day = pd.to_datetime(appoint_day)
            if delta > 0:
                date = self.calendar[self.calendar["trade_date"] < appoint_day]["trade_date"].iloc[-delta]
            if delta < 0:
                date = self.calendar[self.calendar["trade_date"] >= appoint_day]["trade_date"].iloc[-delta]
            if delta == 0:
                date = pd.to_datetime(appoint_day)
            return date
        except Exception as err:
            print("计算指定交易日日期失败：%s偏移%s，错误：%s" % (appoint_day, delta, err))
            return -1

    def day_of_period_end(self, period="W", period_delta=0, day_delta=None):
        try:
            time_con = self.calendar["from_now"] <= 0
            if period == "W":
                week = self.calendar[time_con]["week"].iloc[-1]
                con = self.calendar["week"] == (week - period_delta)
            else:
                month = self.calendar[time_con]["month"].iloc[-1]
                con = self.calendar["month"] == (month - period_delta)
            date = self.calendar[con]["trade_date"].iloc[-1]
            if day_delta:
                date = self.cal_day_delta(date, day_delta)
            return date
        except Exception as err:
            print("计算%s末失败：%s" % (period, err))
            return -1

    def get_date_by_rule(self, rule):
        sell_date = -1
        if isinstance(rule, int):
            sell_date = self.cal_day_delta(self.today, -abs(rule) + 1)
        elif isinstance(rule, list):
            day_delta = None
            if len(rule) == 3:
                day_delta = -abs(rule[2])
            sell_date = self.day_of_period_end(rule[0], -abs(rule[1]), day_delta)
        else:
            print("获取卖出日期失败：%s" % rule)
        return sell_date


class Lunar(object):
    """农历与节气计算，保留原脚本依赖的节假日识别能力。"""

    g_lunar_month_day = [
        0x4AE0, 0xA570, 0x5268, 0xD260, 0xD950, 0x6AA8, 0x56A0, 0x9AD0, 0x4AE8, 0x4AE0,
        0xA4D8, 0xA4D0, 0xD250, 0xD548, 0xB550, 0x56A0, 0x96D0, 0x95B0, 0x49B8, 0x49B0,
        0xA4B0, 0xB258, 0x6A50, 0x6D40, 0xADA8, 0x2B60, 0x9570, 0x4978, 0x4970, 0x64B0,
        0xD4A0, 0xEA50, 0x6D48, 0x5AD0, 0x2B60, 0x9370, 0x92E0, 0xC968, 0xC950, 0xD4A0,
        0xDA50, 0xB550, 0x56A0, 0xAAD8, 0x25D0, 0x92D0, 0xC958, 0xA950, 0xB4A8, 0x6CA0,
        0xB550, 0x55A8, 0x4DA0, 0xA5B0, 0x52B8, 0x52B0, 0xA950, 0xE950, 0x6AA0, 0xAD50,
        0xAB50, 0x4B60, 0xA570, 0xA570, 0x5260, 0xE930, 0xD950, 0x5AA8, 0x56A0, 0x96D0,
        0x4AE8, 0x4AD0, 0xA4D0, 0xD268, 0xD250, 0xD528, 0xB540, 0xB6A0, 0x96D0, 0x95B0,
        0x49B0, 0xA4B8, 0xA4B0, 0xB258, 0x6A50, 0x6D40, 0xADA0, 0xAB60, 0x9370, 0x4978,
        0x4970, 0x64B0, 0x6A50, 0xEA50, 0x6B28, 0x5AC0, 0xAB60, 0x9368, 0x92E0, 0xC960,
        0xD4A8, 0xD4A0, 0xDA50, 0x5AA8, 0x56A0, 0xAAD8, 0x25D0, 0x92D0, 0xC958, 0xA950,
        0xB4A0, 0xB550, 0xB550, 0x55A8, 0x4BA0, 0xA5B0, 0x52B8, 0x52B0, 0xA930, 0x74A8,
        0x6AA0, 0xAD50, 0x4DA8, 0x4B60, 0x9570, 0xA4E0, 0xD260, 0xE930, 0xD530, 0x5AA0,
        0x6B50, 0x96D0, 0x4AE8, 0x4AD0, 0xA4D0, 0xD258, 0xD250, 0xD520, 0xDAA0, 0xB5A0,
        0x56D0, 0x4AD8, 0x49B0, 0xA4B8, 0xA4B0, 0xAA50, 0xB528, 0x6D20, 0xADA0, 0x55B0,
    ]

    g_lunar_month = [
        0x00, 0x50, 0x04, 0x00, 0x20,
        0x60, 0x05, 0x00, 0x20, 0x70,
        0x05, 0x00, 0x40, 0x02, 0x06,
        0x00, 0x50, 0x03, 0x07, 0x00,
        0x60, 0x04, 0x00, 0x20, 0x70,
        0x05, 0x00, 0x30, 0x80, 0x06,
        0x00, 0x40, 0x03, 0x07, 0x00,
        0x50, 0x04, 0x08, 0x00, 0x60,
        0x04, 0x0A, 0x00, 0x60, 0x05,
        0x00, 0x30, 0x80, 0x05, 0x00,
        0x40, 0x02, 0x07, 0x00, 0x50,
        0x04, 0x09, 0x00, 0x60, 0x04,
        0x00, 0x20, 0x60, 0x05, 0x00,
        0x30, 0xB0, 0x06, 0x00, 0x50,
        0x02, 0x07, 0x00, 0x50, 0x03,
    ]

    START_YEAR = 1901
    gan = "甲乙丙丁戊己庚辛壬癸"
    zhi = "子丑寅卯辰巳午未申酉戌亥"
    xiao = "鼠牛虎兔龙蛇马羊猴鸡狗猪"
    lm = "正二三四五六七八九十冬腊"
    ld = "初一初二初三初四初五初六初七初八初九初十十一十二十三十四十五十六十七十八十九二十廿一廿二廿三廿四廿五廿六廿七廿八廿九三十"
    jie = "小寒大寒立春雨水惊蛰春分清明谷雨立夏小满芒种夏至小暑大暑立秋处暑白露秋分寒露霜降立冬小雪大雪冬至"

    def __init__(self, dt=None):
        self.localtime = dt if dt else datetime.datetime.today()

    def sx_year(self):
        year = self.ln_year() - 3 - 1
        year = year % 12
        return self.xiao[year]

    def gz_year(self):
        year = self.ln_year() - 3 - 1
        g = year % 10
        z = year % 12
        return self.gan[g] + self.zhi[z]

    def gz_month(self):
        pass

    def gz_day(self):
        ct = self.localtime
        c = ct.year // 100
        y = ct.year % 100
        y = y - 1 if ct.month == 1 or ct.month == 2 else y
        m = ct.month
        m = m + 12 if ct.month == 1 or ct.month == 2 else m
        d = ct.day
        i = 0 if ct.month % 2 == 1 else 6

        g = 4 * c + c // 4 + 5 * y + y // 4 + 3 * (m + 1) // 5 + d - 3 - 1
        g = g % 10
        z = 8 * c + c // 4 + 5 * y + y // 4 + 3 * (m + 1) // 5 + d + 7 + i - 1
        z = z % 12
        return self.gan[g] + self.zhi[z]

    def gz_hour(self):
        ct = self.localtime
        z = round((ct.hour / 2) + 0.1) % 12
        return self.zhi[z]

    def ln_year(self):
        year, _, _ = self.ln_date()
        return year

    def ln_month(self):
        _, month, _ = self.ln_date()
        return month

    def ln_day(self):
        _, _, day = self.ln_date()
        return day

    def ln_date(self):
        delta_days = self._date_diff()

        if delta_days < 49:
            year = self.START_YEAR - 1
            if delta_days < 19:
                month = 11
                day = 11 + delta_days
            else:
                month = 12
                day = delta_days - 18
            return year, month, day

        delta_days -= 49
        year, month, day = self.START_YEAR, 1, 1
        tmp = self._lunar_year_days(year)
        while delta_days >= tmp:
            delta_days -= tmp
            year += 1
            tmp = self._lunar_year_days(year)

        _, tmp = self._lunar_month_days(year, month)
        while delta_days >= tmp:
            delta_days -= tmp
            if month == self._get_leap_month(year):
                tmp, _ = self._lunar_month_days(year, month)
                if delta_days < tmp:
                    return 0, 0, 0
                delta_days -= tmp
            month += 1
            _, tmp = self._lunar_month_days(year, month)

        day += delta_days
        return year, month, day

    def ln_date_str(self):
        _, month, day = self.ln_date()
        return "{}月{}".format(self.lm[month - 1], self.ld[(day - 1) * 2 : day * 2])

    def ln_jie(self):
        ct = self.localtime
        year = ct.year
        for i in range(24):
            delta = self._julian_day() - self._julian_day_of_ln_jie(year, i)
            if -0.5 <= delta <= 0.5:
                return self.jie[i * 2 : (i + 1) * 2]
        return ""

    def calendar(self):
        pass

    def _date_diff(self):
        return (self.localtime - datetime.datetime(1901, 1, 1)).days

    def _get_leap_month(self, lunar_year):
        flag = self.g_lunar_month[(lunar_year - self.START_YEAR) // 2]
        if (lunar_year - self.START_YEAR) % 2:
            return flag & 0x0F
        return flag >> 4

    def _lunar_month_days(self, lunar_year, lunar_month):
        if lunar_year < self.START_YEAR:
            return 30, 29

        high, low = 0, 29
        ibit = 16 - lunar_month

        if lunar_month > self._get_leap_month(lunar_year) and self._get_leap_month(lunar_year):
            ibit -= 1

        if self.g_lunar_month_day[lunar_year - self.START_YEAR] & (1 << ibit):
            low += 1

        if lunar_month == self._get_leap_month(lunar_year):
            if self.g_lunar_month_day[lunar_year - self.START_YEAR] & (1 << (ibit - 1)):
                high = 30
            else:
                high = 29

        return high, low

    def _lunar_year_days(self, year):
        days = 0
        for i in range(1, 13):
            high, low = self._lunar_month_days(year, i)
            days += high
            days += low
        return days

    def _julian_day(self):
        ct = self.localtime
        year = ct.year
        month = ct.month
        day = ct.day

        if month <= 2:
            month += 12
            year -= 1

        b = year / 100
        b = 2 - b + year / 400

        dd = day + 0.5000115740
        return int(365.25 * (year + 4716) + 0.01) + int(30.60001 * (month + 1)) + dd + b - 1524.5

    def _julian_day_of_ln_jie(self, year, st):
        s_st_acc_info = [
            0.00, 1272494.40, 2548020.60, 3830143.80, 5120226.60, 6420865.80,
            7732018.80, 9055272.60, 10388958.00, 11733065.40, 13084292.40, 14441592.00,
            15800560.80, 17159347.20, 18513766.20, 19862002.20, 21201005.40, 22529659.80,
            23846845.20, 25152606.00, 26447687.40, 27733451.40, 29011921.20, 30285477.60,
        ]

        base1900_slight_cold_jd = 2415025.5868055555

        if (st < 0) or (st > 24):
            return 0.0

        st_jd = 365.24219878 * (year - 1900) + s_st_acc_info[st] / 86400.0
        return base1900_slight_cold_jd + st_jd


def load_eod_dates(eod_path: Path) -> list[str]:
    """读取 EOD 交易日。脚本只依赖这个文件。"""
    dates_path = Path(eod_path) / "dates.npy"
    if not dates_path.exists():
        raise FileNotFoundError(f"dates.npy not found: {dates_path}")
    return [normalize_date_label(x) for x in decode_array(np.load(dates_path, allow_pickle=True))]


def build_period_id_frame(
    dates: list[str],
    period_dict: dict[Any, list[Any]] | None = None,
    min_day: int = DEFAULT_MIN_DAY,
) -> pd.DataFrame:
    """根据 EOD 交易日生成各周期 offset 的持仓周期编号宽表。"""
    period_dict = period_dict or DEFAULT_PERIOD_DICT
    index_data = pd.DataFrame({"交易日期": pd.to_datetime(dates, format="%Y%m%d")})
    index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)
    raw = calc_period_and_offset(period_dict, index_data, end_day=index_data["交易日期"].iloc[-1], min_day=min_day)
    return align_period_frame(raw, index_data["交易日期"])


def calc_period_and_offset(
    period_dict: dict[Any, list[Any]],
    _index_data: pd.DataFrame,
    end_day,
    min_day: int = DEFAULT_MIN_DAY,
) -> pd.DataFrame:
    """按原脚本 calc_period_and_offset 的周期逻辑生成周期编号宽表。"""
    _index_data = _index_data.copy()
    _index_data["交易日期"] = pd.to_datetime(_index_data["交易日期"])
    _index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)

    # 原脚本会在尾部补一段未来交易日，避免最后一个周期因样本截断被错误切碎。
    lastday = _index_data["交易日期"].iloc[-1]
    day_win = max((pd.to_datetime(end_day) - lastday).days, 50)
    tc = TradeCalendar(day_win)
    calendar_data = tc.calendar[tc.calendar["trade_date"] > lastday].copy()
    if pd.to_datetime(end_day) > lastday + pd.to_timedelta("50d"):
        calendar_data = calendar_data[calendar_data["trade_date"] <= pd.to_datetime(end_day)]
    calendar_data.rename(columns={"trade_date": "交易日期"}, inplace=True)
    calendar_data = calendar_data[["交易日期"]]
    _index_data = pd.concat([_index_data, calendar_data], ignore_index=True)
    _index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)

    _index_data["是否交易"] = 1
    _index_data["周期最后交易日"] = _index_data["交易日期"]
    all_period_offst_df = _index_data[["交易日期"]].copy()

    agg_dict = {"周期最后交易日": "last", "是否交易": "sum"}

    for period_type in period_dict:
        if period_type in ["W53"]:
            index_data = _index_data.copy()
            start_date = index_data["交易日期"].min()
            end_date = index_data["交易日期"].max()

            index_data["交易日期"] -= pd.to_timedelta("3D")
            index_data.loc[index_data.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
            index_data.set_index("交易日期", inplace=True)
            period_df = index_data.resample(rule="W").agg(agg_dict)
            period_df.rename(columns={"是否交易": "交易天数"}, inplace=True)
            period_df = period_df[period_df["交易天数"] > 0]
            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={"周期最后交易日": "交易日期"}, inplace=True)
            period_df["W53"] = 1

            df = pd.merge(
                _index_data[["交易日期"]].copy(),
                right=period_df[["交易日期", "W53"]],
                on="交易日期",
                how="left",
            )
            df["W53_0"] = df["W53"].expanding().sum().shift()

            date_range_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq="D"), columns=["交易日期"])
            date_range_df["周期最后交易日"] = date_range_df["交易日期"].copy()
            date_range_df["交易日期"] -= pd.to_timedelta("4D")
            date_range_df.loc[date_range_df.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
            date_range_df.set_index("交易日期", inplace=True)
            period_df = date_range_df.resample(rule="W").agg({"周期最后交易日": "last"})
            df.loc[df["交易日期"].isin(period_df["周期最后交易日"]), "W53_0"] = None

            counts = df["W53_0"].value_counts()
            single_occurrence_values = counts[counts == 1].index
            df.loc[df["W53_0"].isin(single_occurrence_values), "W53_0"] = None

            df["_W53"] = df["W53_0"].copy()
            df["W53_0"] = df["W53_0"].ffill()
            df.loc[pd.isnull(df["_W53"]), "W53_0"] = -df["W53_0"]
            all_period_offst_df = pd.merge(
                left=all_period_offst_df,
                right=df[["交易日期", "W53_0"]],
                on="交易日期",
                how="left",
            )
            continue

        for offset in period_dict[period_type]:
            index_data = _index_data.copy()
            period_key = f"{period_type}_{offset}"
            if type(period_type) == int:
                index_data["group"] = pd.Series((index_data.index - offset) / period_type).apply(int)
                period_df = index_data.groupby("group").agg(agg_dict)
                period_df["交易天数"] = period_type
            else:
                if (period_type == "M") and (offset == -5):
                    start_date = index_data["交易日期"].min()
                    end_date = index_data["交易日期"].max()
                    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
                    index_data = index_data.set_index("交易日期").reindex(date_range).reset_index()
                    index_data.rename(columns={"index": "交易日期"}, inplace=True)
                    index_data["周期最后交易日"] = index_data["周期最后交易日"].ffill()
                    index_data["是否交易"] = index_data["是否交易"].fillna(value=0)

                    date_range_m = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq="M"), columns=["交易日期"])
                    date_range_m["月末"] = 1
                    index_data = pd.merge(left=index_data, right=date_range_m, on="交易日期", how="left")
                    index_data.loc[(index_data["月末"] == 1) & (index_data["是否交易"].shift(-1) == 0), "是否交易"] = 0
                    index_data = index_data[index_data["是否交易"] == 0]
                    index_data.set_index("交易日期", inplace=True)
                    period_df = index_data.resample(rule=period_type).agg(agg_dict)
                    period_df["交易天数"] = 20
                else:
                    if (lambda s: any(char.isdigit() for char in s))(period_type):
                        if isinstance(offset, str) and "D" in offset.upper():
                            offset = offset.upper()
                            index_data["交易日期"] -= pd.to_timedelta(offset)
                        else:
                            index_data["交易日期"] -= pd.to_timedelta(f"{offset * 7}D")
                    else:
                        index_data["交易日期"] -= pd.to_timedelta(f"{offset}D")

                    index_data.loc[index_data.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
                    index_data.set_index("交易日期", inplace=True)
                    period_df = index_data.resample(rule=period_type).agg(agg_dict)
                    period_df.rename(columns={"是否交易": "交易天数"}, inplace=True)

            period_df = period_df[period_df["交易天数"] > 0]

            # min_day 是给周/月频和多日周期处理节假日短周期用的。
            # 1 日换仓天然每期只有 1 个交易日，不能套用 min_day=2，否则 1_0 会被全部过滤掉。
            if not (type(period_type) == int and period_type == 1):
                index_to_remove = []
                add_num = 0
                for index, row in period_df.iterrows():
                    period_df.at[index, "交易天数"] += add_num
                    add_num = 0
                    if row["交易天数"] < min_day:
                        index_to_remove.append(index)
                        add_num = row["交易天数"]
                period_df = period_df.drop(index_to_remove)

            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={"周期最后交易日": "交易日期"}, inplace=True)
            period_df[period_key] = 1
            all_period_offst_df = pd.merge(
                left=all_period_offst_df,
                right=period_df[["交易日期", period_key]],
                on="交易日期",
                how="left",
            )
            all_period_offst_df[period_key] = all_period_offst_df[period_key].expanding().sum().shift()

    all_period_offst_df.fillna(value=0, inplace=True)
    return all_period_offst_df


def align_period_frame(raw: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    """裁剪并对齐回 eod/dates.npy 的日期，保证矩阵一日一列。"""
    out = raw.copy()
    out["交易日期"] = pd.to_datetime(out["交易日期"])
    out = out.drop_duplicates(subset=["交易日期"], keep="last")
    aligned = pd.DataFrame({"交易日期": pd.to_datetime(target_dates)})
    aligned = aligned.merge(out, on="交易日期", how="left")
    aligned.fillna(value=0, inplace=True)
    return aligned.rename(columns={"交易日期": "trade_date"})


def build_period_rebalance_matrix(
    dates: list[str],
    period_dict: dict[Any, list[Any]] | None = None,
    min_day: int = DEFAULT_MIN_DAY,
) -> tuple[list[str], np.ndarray]:
    """把周期编号宽表转换成换仓日布尔矩阵。"""
    ids = build_period_id_frame(dates, period_dict=period_dict, min_day=min_day)
    period_names = [col for col in ids.columns if col != "trade_date"]
    values = ids[period_names].to_numpy(dtype=float)
    previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
    rebalance = (values != previous) & (values != 0)
    if rebalance.shape[0] > 0:
        rebalance[0, :] = False
    return period_names, rebalance.T.astype(bool, copy=False)


def save_period_files(
    eod_path: Path = DEFAULT_EOD_PATH,
    period_path: Path = DEFAULT_PERIOD_PATH,
    period_dict: dict[Any, list[Any]] | None = None,
    min_day: int = DEFAULT_MIN_DAY,
) -> tuple[Path, Path, Path]:
    """从 eod/dates.npy 生成 period.npy、period_dates.npy、dates.npy。"""
    eod_path = Path(eod_path)
    period_path = Path(period_path)
    period_path.mkdir(parents=True, exist_ok=True)
    dates = load_eod_dates(eod_path)
    period_names, rebalance = build_period_rebalance_matrix(dates, period_dict=period_dict, min_day=min_day)

    names_path = period_path / "period.npy"
    mask_path = period_path / "period_dates.npy"
    dates_path = period_path / "dates.npy"
    np.save(names_path, np.asarray(period_names, dtype=object))
    np.save(mask_path, rebalance)
    np.save(dates_path, np.asarray(dates, dtype="S8"))
    return names_path, mask_path, dates_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone period rebalance npy files from eod/dates.npy")
    parser.add_argument("--eod-path", default=str(DEFAULT_EOD_PATH), help="EOD directory containing dates.npy")
    parser.add_argument("--period-path", default=str(DEFAULT_PERIOD_PATH), help="Output directory for period npy files")
    parser.add_argument("--min-day", type=int, default=DEFAULT_MIN_DAY, help="Minimum trading days in one period")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names_path, mask_path, dates_path = save_period_files(
        eod_path=Path(args.eod_path),
        period_path=Path(args.period_path),
        min_day=int(args.min_day),
    )
    print(f"saved: {names_path}")
    print(f"saved: {mask_path}")
    print(f"saved: {dates_path}")


if __name__ == "__main__":
    main()
