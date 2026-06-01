# -*- coding: utf-8 -*-
"""周期预运算工具。

本模块已经把 ``周期预运算（沈博文增强版）.py`` 中 FactorForge 需要的周期
生成逻辑搬进来，后续不再依赖那个独立脚本。唯一的输入变化是：
原脚本从指数 CSV 获取交易日，这里从 ``eod/dates.npy`` 获取交易日。

输出文件：
- ``period.npy``：周期名称，例如 ``5_0``、``20_19``、``W_4``。
- ``period_dates.npy``：``period x dates`` 的 True/False 换仓日矩阵。
- ``dates.npy``：与 EOD 对齐的日期标签。
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import normalize_offset_label, normalize_period
from .data_loader import decode_array, normalize_date_label


# 交易日历，离线版本，几乎同Rocket，仅改为离线版
class TradeCalendar(object):
    # 常用的参数，初始化的时候就会生成
    today = None  # 当前交易日期，如果当前非交易日。
    last_trade_date = None  # 上一个交易日
    next_trade_date = None  # 下一个交易日
    market_open = True  # 今日是否交易
    calendar = None  # 完整的交易日历
    _now_date = datetime.date.today()

    def __init__(self, day_win=60):
        """
        构造函数
        :param day_win: 日历长度
        """

        self.calendar = self.create_calendar_offline(day_win=day_win * 1.5)
        # 计算时间差
        self.calendar['from_now'] = self.calendar['trade_date'].apply(lambda x: (x - self._now_date).days)
        # 只保留距指定时间的数据
        # self.calendar = self.calendar[abs(self.calendar['from_now']) < day_win]
        self.calendar = self.calendar[self.calendar['from_now'] < day_win]
        # 处理AKshare获取的未来日历长度不够的问题
        if self.calendar['from_now'].max() < day_win * 0.95:
            calendar_offline = self.create_calendar_offline(day_win=day_win * 1.5)
            calendar_offline = calendar_offline[calendar_offline['trade_date'] > self.calendar['trade_date'].iloc[-1]]
            self.calendar = pd.concat([self.calendar, calendar_offline], ignore_index=True).sort_values(
                by='trade_date')
            self.calendar['from_now'] = self.calendar['trade_date'].apply(lambda x: (x - self._now_date).days)

        self.calendar['trade_date'] = pd.to_datetime(self.calendar['trade_date'])
        # 获取上一个交易日
        con = self.calendar['from_now'] < 0
        self.last_trade_date = self.calendar[con]['trade_date'].iloc[-1]

        # 获取下一个交易日
        con = self.calendar['from_now'] > 0
        self.next_trade_date = self.calendar[con]['trade_date'].iloc[0]

        # 获取当前日期
        if 0 in self.calendar['from_now'].values:
            self.today = self.calendar[self.calendar['from_now'] == 0]['trade_date'].iloc[0]
        else:
            self.today = self.next_trade_date
            self.market_open = False

        # 计算好一些东西备用
        self.calendar['month'] = self.calendar['trade_date'].apply(lambda x: x.month)
        self.calendar['week'] = self.calendar['trade_date'].apply(lambda x: x.weekofyear)
        for key in ['month', 'week']:
            self.calendar.loc[self.calendar[key] == self.calendar[key].shift(), key] = 0
            self.calendar.loc[self.calendar[key] != 0, key] = 1
            self.calendar[key] = self.calendar[key].cumsum()
        # 计算当前日属于一周的第几天或者一个月的第几天
        self.calendar['week_of'] = self.calendar.groupby('week')['week'].rank(method='first')
        self.calendar['month_of'] = self.calendar.groupby('month')['month'].rank(method='first')

    @staticmethod
    def create_calendar_offline(day_win=84):
        """
        离线创建交易日历
        :param day_win: 日历长度
        :return:
        """
        now_date = datetime.date.today()
        start = now_date - datetime.timedelta(days=day_win)
        end = now_date + datetime.timedelta(days=day_win)
        calendar_offline = pd.DataFrame(pd.date_range(start, end, freq='D'))
        calendar_offline.rename(columns={0: 'trade_date'}, inplace=True)
        # ===== 删除节假日数据的数据
        # 1、标记周末、和日期信息：公历+阴历+节气
        calendar_offline['week'] = calendar_offline['trade_date'].dt.dayofweek + 1
        calendar_offline['date_info'] = calendar_offline['trade_date'].apply(
            lambda x: '公历%s月%s日' % (x.month, x.day) + ' 农历' + Lunar(x).ln_date_str() + Lunar(x).ln_jie())
        # 节日放假信息
        holidays = {'公历1月1日': 3, '公历5月1日': 5, '公历10月1日': 7,
                    '正月初一': 6, '五月初五': 3, '八月十五': 3,
                    '清明': 3}
        for holiday in holidays.keys():
            calendar_offline.loc[calendar_offline['date_info'].str.contains(holiday), 'rest'] = holidays[holiday]

        # 周末默认放假
        calendar_offline.loc[calendar_offline['week'] > 5, 'holiday'] = 1
        # 根据节假日的休息信息判断什么时候放假
        for i in calendar_offline.index[10:-10]:
            if calendar_offline.at[i, 'rest'] > 0:
                # 节日当天肯定是休息的
                calendar_offline.at[i, 'holiday'] = 1
                week = calendar_offline.at[i, 'week']
                rest = calendar_offline.at[i, 'rest']
                # 处理休息三天的情况
                if rest == 3:
                    if week <= 2:
                        calendar_offline.at[i - 1, 'holiday'] = 1
                        calendar_offline.at[i - 2, 'holiday'] = 1
                    elif (week > 3) & (week < 7):
                        calendar_offline.at[i + 1, 'holiday'] = 1
                        calendar_offline.at[i + 2, 'holiday'] = 1
                    else:
                        calendar_offline.at[i + 1, 'holiday'] = 1
                else:
                    for j in range(1, int(rest)):
                        calendar_offline.at[i + j, 'holiday'] = 1
                    # 特殊处理春节,春节有是29开始放假，有的是30开始放假。
                    if rest == 6:
                        calendar_offline.at[i - 1, 'holiday'] = 1

        calendar_offline = calendar_offline[calendar_offline['holiday'] != 1][['trade_date']]
        calendar_offline['trade_date'] = calendar_offline['trade_date'].apply(
            lambda x: datetime.date(year=x.year, month=x.month, day=x.day))
        return calendar_offline

    def cal_day_delta(self, appoint_day, delta):
        """
        根据指定日期计算日期偏差。
        :param appoint_day: 指定日期
        :param delta: 日期偏差，正数表示向前，负数表示向后
        :return:
        """
        try:
            # 时间格式转换
            if isinstance(appoint_day, str):
                appoint_day = pd.to_datetime(appoint_day)
            if delta > 0:
                date = self.calendar[self.calendar['trade_date'] < appoint_day]['trade_date'].iloc[-delta]
            if delta < 0:
                date = self.calendar[self.calendar['trade_date'] >= appoint_day]['trade_date'].iloc[-delta]
            if delta == 0:
                date = pd.to_datetime(appoint_day)
            return date
        except Exception as err:
            print('计算指定交易日日期失败：%s偏移%s，错误：%s' % (appoint_day, delta, err))
            return -1

    def day_of_period_end(self, period='W', period_delta=0, day_delta=None):
        """
        计算周末或者月末的交易日期
        :param period: 周还是月
        :param period_delta: 间隔周或者月，-1表示下周，0表示当周，1表示上周
        :param day_delta: 针对获取到的末期数据计算便宜
        :return:
        """
        try:
            time_con = self.calendar['from_now'] <= 0
            if period == 'W':
                week = self.calendar[time_con]['week'].iloc[-1]
                con = self.calendar['week'] == (week - period_delta)
            else:
                month = self.calendar[time_con]['month'].iloc[-1]
                con = self.calendar['month'] == (month - period_delta)
            date = self.calendar[con]['trade_date'].iloc[-1]
            if day_delta:
                date = self.cal_day_delta(date, day_delta)
            return date
        except Exception as err:
            print('计算%s末失败：%s' % (period, err))
            return -1

    def get_date_by_rule(self, rule):
        """
        根据规则获取指定的交易日期，详见使用指南
        :param rule: 规则
        :return:
        """
        sell_date = -1
        if isinstance(rule, int):
            sell_date = self.cal_day_delta(self.today, -abs(rule) + 1)
        elif isinstance(rule, list):
            day_delta = None
            if len(rule) == 3:
                day_delta = -abs(rule[2])
            sell_date = self.day_of_period_end(rule[0], -abs(rule[1]), day_delta)
        else:
            print('获取卖出日期失败：%s' % rule)
        return sell_date


# 计算农历节假日，copy的代码，从Rocket抄来
class Lunar(object):
    # ******************************************************************************
    # 下面为阴历计算所需的数据,为节省存储空间,所以采用下面比较变态的存储方法.
    # ******************************************************************************
    # 数组g_lunar_month_day存入阴历1901年到2050年每年中的月天数信息，
    # 阴历每月只能是29或30天，一年用12（或13）个二进制位表示，对应位为1表30天，否则为29天
    g_lunar_month_day = [
        0x4ae0, 0xa570, 0x5268, 0xd260, 0xd950, 0x6aa8, 0x56a0, 0x9ad0, 0x4ae8, 0x4ae0,  # 1910
        0xa4d8, 0xa4d0, 0xd250, 0xd548, 0xb550, 0x56a0, 0x96d0, 0x95b0, 0x49b8, 0x49b0,  # 1920
        0xa4b0, 0xb258, 0x6a50, 0x6d40, 0xada8, 0x2b60, 0x9570, 0x4978, 0x4970, 0x64b0,  # 1930
        0xd4a0, 0xea50, 0x6d48, 0x5ad0, 0x2b60, 0x9370, 0x92e0, 0xc968, 0xc950, 0xd4a0,  # 1940
        0xda50, 0xb550, 0x56a0, 0xaad8, 0x25d0, 0x92d0, 0xc958, 0xa950, 0xb4a8, 0x6ca0,  # 1950
        0xb550, 0x55a8, 0x4da0, 0xa5b0, 0x52b8, 0x52b0, 0xa950, 0xe950, 0x6aa0, 0xad50,  # 1960
        0xab50, 0x4b60, 0xa570, 0xa570, 0x5260, 0xe930, 0xd950, 0x5aa8, 0x56a0, 0x96d0,  # 1970
        0x4ae8, 0x4ad0, 0xa4d0, 0xd268, 0xd250, 0xd528, 0xb540, 0xb6a0, 0x96d0, 0x95b0,  # 1980
        0x49b0, 0xa4b8, 0xa4b0, 0xb258, 0x6a50, 0x6d40, 0xada0, 0xab60, 0x9370, 0x4978,  # 1990
        0x4970, 0x64b0, 0x6a50, 0xea50, 0x6b28, 0x5ac0, 0xab60, 0x9368, 0x92e0, 0xc960,  # 2000
        0xd4a8, 0xd4a0, 0xda50, 0x5aa8, 0x56a0, 0xaad8, 0x25d0, 0x92d0, 0xc958, 0xa950,  # 2010
        0xb4a0, 0xb550, 0xb550, 0x55a8, 0x4ba0, 0xa5b0, 0x52b8, 0x52b0, 0xa930, 0x74a8,  # 2020
        0x6aa0, 0xad50, 0x4da8, 0x4b60, 0x9570, 0xa4e0, 0xd260, 0xe930, 0xd530, 0x5aa0,  # 2030
        0x6b50, 0x96d0, 0x4ae8, 0x4ad0, 0xa4d0, 0xd258, 0xd250, 0xd520, 0xdaa0, 0xb5a0,  # 2040
        0x56d0, 0x4ad8, 0x49b0, 0xa4b8, 0xa4b0, 0xaa50, 0xb528, 0x6d20, 0xada0, 0x55b0,  # 2050
    ]

    # 数组gLanarMonth存放阴历1901年到2050年闰月的月份，如没有则为0，每字节存两年
    g_lunar_month = [
        0x00, 0x50, 0x04, 0x00, 0x20,  # 1910
        0x60, 0x05, 0x00, 0x20, 0x70,  # 1920
        0x05, 0x00, 0x40, 0x02, 0x06,  # 1930
        0x00, 0x50, 0x03, 0x07, 0x00,  # 1940
        0x60, 0x04, 0x00, 0x20, 0x70,  # 1950
        0x05, 0x00, 0x30, 0x80, 0x06,  # 1960
        0x00, 0x40, 0x03, 0x07, 0x00,  # 1970
        0x50, 0x04, 0x08, 0x00, 0x60,  # 1980
        0x04, 0x0a, 0x00, 0x60, 0x05,  # 1990
        0x00, 0x30, 0x80, 0x05, 0x00,  # 2000
        0x40, 0x02, 0x07, 0x00, 0x50,  # 2010
        0x04, 0x09, 0x00, 0x60, 0x04,  # 2020
        0x00, 0x20, 0x60, 0x05, 0x00,  # 2030
        0x30, 0xb0, 0x06, 0x00, 0x50,  # 2040
        0x02, 0x07, 0x00, 0x50, 0x03  # 2050
    ]

    START_YEAR = 1901

    # 天干
    gan = '甲乙丙丁戊己庚辛壬癸'
    # 地支
    zhi = '子丑寅卯辰巳午未申酉戌亥'
    # 生肖
    xiao = '鼠牛虎兔龙蛇马羊猴鸡狗猪'
    # 月份
    lm = '正二三四五六七八九十冬腊'
    # 日份
    ld = '初一初二初三初四初五初六初七初八初九初十十一十二十三十四十五十六十七十八十九二十廿一廿二廿三廿四廿五廿六廿七廿八廿九三十'
    # 节气
    jie = '小寒大寒立春雨水惊蛰春分清明谷雨立夏小满芒种夏至小暑大暑立秋处暑白露秋分寒露霜降立冬小雪大雪冬至'

    def __init__(self, dt=None):
        '''初始化：参数为datetime.datetime类实例，默认当前时间'''
        self.localtime = dt if dt else datetime.datetime.today()

    def sx_year(self):  # 返回生肖年
        ct = self.localtime  # 取当前时间

        year = self.ln_year() - 3 - 1  # 农历年份减3 （说明：补减1）
        year = year % 12  # 模12，得到地支数
        return self.xiao[year]

    def gz_year(self):  # 返回干支纪年
        ct = self.localtime  # 取当前时间
        year = self.ln_year() - 3 - 1  # 农历年份减3 （说明：补减1）
        G = year % 10  # 模10，得到天干数
        Z = year % 12  # 模12，得到地支数
        return self.gan[G] + self.zhi[Z]

    def gz_month(self):  # 返回干支纪月（未实现）
        pass

    def gz_day(self):  # 返回干支纪日
        ct = self.localtime  # 取当前时间
        C = ct.year // 100  # 取世纪数，减一
        y = ct.year % 100  # 取年份后两位（若为1月、2月则当前年份减一）
        y = y - 1 if ct.month == 1 or ct.month == 2 else y
        M = ct.month  # 取月份（若为1月、2月则分别按13、14来计算）
        M = M + 12 if ct.month == 1 or ct.month == 2 else M
        d = ct.day  # 取日数
        i = 0 if ct.month % 2 == 1 else 6  # 取i （奇数月i=0，偶数月i=6）

        # 下面两个是网上的公式
        # http://baike.baidu.com/link?url=MbTKmhrTHTOAz735gi37tEtwd29zqE9GJ92cZQZd0X8uFO5XgmyMKQru6aetzcGadqekzKd3nZHVS99rewya6q
        # 计算干（说明：补减1）
        G = 4 * C + C // 4 + 5 * y + y // 4 + 3 * (M + 1) // 5 + d - 3 - 1
        G = G % 10
        # 计算支（说明：补减1）
        Z = 8 * C + C // 4 + 5 * y + y // 4 + 3 * (M + 1) // 5 + d + 7 + i - 1
        Z = Z % 12

        # 返回 干支纪日
        return self.gan[G] + self.zhi[Z]

    def gz_hour(self):  # 返回干支纪时（时辰）
        ct = self.localtime  # 取当前时间
        # 计算支
        Z = round((ct.hour / 2) + 0.1) % 12  # 之所以加0.1是因为round的bug!!

        # 返回 干支纪时（时辰）
        return self.zhi[Z]

    def ln_year(self):  # 返回农历年
        year, _, _ = self.ln_date()
        return year

    def ln_month(self):  # 返回农历月
        _, month, _ = self.ln_date()
        return month

    def ln_day(self):  # 返回农历日
        _, _, day = self.ln_date()
        return day

    def ln_date(self):  # 返回农历日期整数元组（年、月、日）（查表法）
        delta_days = self._date_diff()

        # 阳历1901年2月19日为阴历1901年正月初一
        # 阳历1901年1月1日到2月19日共有49天
        if (delta_days < 49):
            year = self.START_YEAR - 1
            if (delta_days < 19):
                month = 11;
                day = 11 + delta_days
            else:
                month = 12;
                day = delta_days - 18
            return (year, month, day)

        # 下面从阴历1901年正月初一算起
        delta_days -= 49
        year, month, day = self.START_YEAR, 1, 1
        # 计算年
        tmp = self._lunar_year_days(year)
        while delta_days >= tmp:
            delta_days -= tmp
            year += 1
            tmp = self._lunar_year_days(year)

        # 计算月
        (foo, tmp) = self._lunar_month_days(year, month)
        while delta_days >= tmp:
            delta_days -= tmp
            if (month == self._get_leap_month(year)):
                (tmp, foo) = self._lunar_month_days(year, month)
                if (delta_days < tmp):
                    return (0, 0, 0)
                delta_days -= tmp
            month += 1
            (foo, tmp) = self._lunar_month_days(year, month)

        # 计算日
        day += delta_days
        return (year, month, day)

    def ln_date_str(self):  # 返回农历日期字符串，形如：农历正月初九
        _, month, day = self.ln_date()
        return '{}月{}'.format(self.lm[month - 1], self.ld[(day - 1) * 2:day * 2])

    def ln_jie(self):  # 返回农历节气
        ct = self.localtime  # 取当前时间
        year = ct.year
        for i in range(24):
            # 因为两个都是浮点数，不能用相等表示
            delta = self._julian_day() - self._julian_day_of_ln_jie(year, i)
            if -.5 <= delta <= .5:
                return self.jie[i * 2:(i + 1) * 2]
        return ''

    # 显示日历
    def calendar(self):
        pass

    #######################################################
    #            下面皆为私有函数
    #######################################################

    def _date_diff(self):
        '''返回基于1901/01/01日差数'''
        return (self.localtime - datetime.datetime(1901, 1, 1)).days

    def _get_leap_month(self, lunar_year):
        flag = self.g_lunar_month[(lunar_year - self.START_YEAR) // 2]
        if (lunar_year - self.START_YEAR) % 2:
            return flag & 0x0f
        else:
            return flag >> 4

    def _lunar_month_days(self, lunar_year, lunar_month):
        if (lunar_year < self.START_YEAR):
            return 30

        high, low = 0, 29
        iBit = 16 - lunar_month;

        if (lunar_month > self._get_leap_month(lunar_year) and self._get_leap_month(lunar_year)):
            iBit -= 1

        if (self.g_lunar_month_day[lunar_year - self.START_YEAR] & (1 << iBit)):
            low += 1

        if (lunar_month == self._get_leap_month(lunar_year)):
            if (self.g_lunar_month_day[lunar_year - self.START_YEAR] & (1 << (iBit - 1))):
                high = 30
            else:
                high = 29

        return (high, low)

    def _lunar_year_days(self, year):
        days = 0
        for i in range(1, 13):
            (high, low) = self._lunar_month_days(year, i)
            days += high
            days += low
        return days

    # 返回指定公历日期的儒略日（http://blog.csdn.net/orbit/article/details/9210413）
    def _julian_day(self):
        ct = self.localtime  # 取当前时间
        year = ct.year
        month = ct.month
        day = ct.day

        if month <= 2:
            month += 12
            year -= 1

        B = year / 100
        B = 2 - B + year / 400

        dd = day + 0.5000115740  # 本日12:00后才是儒略日的开始(过一秒钟)*/
        return int(365.25 * (year + 4716) + 0.01) + int(30.60001 * (month + 1)) + dd + B - 1524.5

    # 返回指定年份的节气的儒略日数（http://blog.csdn.net/orbit/article/details/9210413）
    def _julian_day_of_ln_jie(self, year, st):
        s_stAccInfo = [
            0.00, 1272494.40, 2548020.60, 3830143.80, 5120226.60, 6420865.80,
            7732018.80, 9055272.60, 10388958.00, 11733065.40, 13084292.40, 14441592.00,
            15800560.80, 17159347.20, 18513766.20, 19862002.20, 21201005.40, 22529659.80,
            23846845.20, 25152606.00, 26447687.40, 27733451.40, 29011921.20, 30285477.60]

        # 已知1900年小寒时刻为1月6日02:05:00
        base1900_SlightColdJD = 2415025.5868055555

        if (st < 0) or (st > 24):
            return 0.0

        stJd = 365.24219878 * (year - 1900) + s_stAccInfo[st] / 86400.0

        return base1900_SlightColdJD + stJd


def default_period_dict() -> dict[Any, list[Any]]:
    """返回默认预计算的周期与 offset 集合。"""
    return {
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


def load_period_files(period_path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """读取预计算周期文件，并校验 ``period_dates`` 的二维形状。"""
    period_path = Path(period_path)
    names_path = period_path / "period.npy"
    mask_path = period_path / "period_dates.npy"
    dates_path = period_path / "dates.npy"
    missing = [path for path in [names_path, mask_path, dates_path] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"period files not found: {missing_text}")

    period_names = decode_array(np.load(names_path, allow_pickle=True))
    period_dates = np.load(mask_path, allow_pickle=False)
    dates = [normalize_date_label(x) for x in decode_array(np.load(dates_path, allow_pickle=True))]
    if period_dates.shape != (len(period_names), len(dates)):
        raise ValueError(
            f"period_dates shape {period_dates.shape} does not match "
            f"period count {len(period_names)} and date count {len(dates)}"
        )
    return period_names, period_dates.astype(bool, copy=False), dates


def resolve_period_keys(period_names: list[str], period: Any, offsets: str | list[str]) -> list[str]:
    """把 YAML 中的 period/offsets 配置解析成 ``20_7`` 这样的具体列名。"""
    base = normalize_period(period)
    prefix = f"{base}_"
    available = [name for name in period_names if name.startswith(prefix)]
    if offsets == "all":
        keys = available
    else:
        keys = [format_period_key(base, offset) for offset in offsets]

    missing = [key for key in keys if key not in period_names]
    if missing:
        sample = ", ".join(available[:12])
        raise ValueError(f"period keys not found: {missing}. Available for {base}: {sample}")
    return keys


def format_period_key(period: Any, offset: Any) -> str:
    """按 ``period.npy`` 的命名规则拼出周期键。"""
    return f"{normalize_period(period)}_{normalize_offset_label(offset)}"


def sort_period_keys(keys: list[str]) -> list[str]:
    """保留兼容接口；周期文件本身已经按原脚本生成顺序排列。"""
    return list(keys)


def build_period_id_frame(dates: list[str], period_dict: dict[Any, list[Any]] | None = None, min_day: int = 2) -> pd.DataFrame:
    """根据 EOD 交易日生成各周期 offset 的“持仓周期编号”宽表。"""
    period_dict = period_dict or default_period_dict()
    index_data = pd.DataFrame({"交易日期": pd.to_datetime(dates, format="%Y%m%d")})
    index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)
    raw = calc_period_and_offset(period_dict, index_data, end_day=index_data["交易日期"].iloc[-1], min_day=min_day)
    return _align_period_frame(raw, index_data["交易日期"])


def calc_period_and_offset(
    period_dict: dict[Any, list[Any]],
    _index_data: pd.DataFrame,
    end_day,
    min_day: int = 2,
) -> pd.DataFrame:
    """按原脚本 ``calc_period_and_offset`` 的周期逻辑生成周期编号宽表。

    这里保留原脚本开头用 ``TradeCalendar`` 补未来交易日的节假日逻辑；最终写 npy 前，
    再由 ``_align_period_frame`` 裁剪回 ``eod/dates.npy`` 的日期范围。
    """
    _index_data = _index_data.copy()
    _index_data["交易日期"] = pd.to_datetime(_index_data["交易日期"])
    _index_data.sort_values(by=["交易日期"], ascending=True, ignore_index=True, inplace=True)

    # 以下补未来交易日逻辑来自原脚本：当尾部周期不足时，用离线节假日日历向后补一段，
    # 让短周期合并、周/月频分割在样本尾部也能按原口径处理。
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

            # 第一步：先把 W_3 算一遍，目的和原脚本一致，是把周三周期末定义出来。
            index_data["交易日期"] -= pd.to_timedelta("3D")
            index_data.loc[index_data.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
            index_data.set_index("交易日期", inplace=True)
            period_df = index_data.resample(rule="W").agg(agg_dict)
            period_df.rename(columns={"是否交易": "交易天数"}, inplace=True)
            period_df = period_df[period_df["交易天数"] > 0]
            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={"周期最后交易日": "交易日期"}, inplace=True)
            period_df["W53"] = 1

            df = pd.merge(_index_data[["交易日期"]].copy(), right=period_df[["交易日期", "W53"]], on="交易日期", how="left")
            df["W53_0"] = df["W53"].expanding().sum().shift()

            # 第二步：把所有周四标记为 None。
            date_range_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq="D"), columns=["交易日期"])
            date_range_df["周期最后交易日"] = date_range_df["交易日期"].copy()
            date_range_df["交易日期"] -= pd.to_timedelta("4D")
            date_range_df.loc[date_range_df.index.max() + 1, "交易日期"] = pd.to_datetime("1990-01-01")
            date_range_df.set_index("交易日期", inplace=True)
            period_df = date_range_df.resample(rule="W").agg({"周期最后交易日": "last"})
            df.loc[df["交易日期"].isin(period_df["周期最后交易日"]), "W53_0"] = None

            # 第三步：单交易日周期不满足 T+1，设置为 None。
            counts = df["W53_0"].value_counts()
            single_occurrence_values = counts[counts == 1].index
            df.loc[df["W53_0"].isin(single_occurrence_values), "W53_0"] = None

            # 第四步：None 做 ffill 后取负，保留原脚本的名义周期标记。
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
            period_df[f"{period_type}_{offset}"] = 1
            all_period_offst_df = pd.merge(
                left=all_period_offst_df,
                right=period_df[["交易日期", f"{period_type}_{offset}"]],
                on="交易日期",
                how="left",
            )
            all_period_offst_df[f"{period_type}_{offset}"] = all_period_offst_df[f"{period_type}_{offset}"].expanding().sum().shift()

    all_period_offst_df.fillna(value=0, inplace=True)
    return all_period_offst_df


def _align_period_frame(raw: pd.DataFrame, target_dates: pd.Series) -> pd.DataFrame:
    """把周期编号宽表整理成与 ``eod/dates.npy`` 一日一行的输出。"""
    out = raw.copy()
    out["交易日期"] = pd.to_datetime(out["交易日期"])
    # M_-5 在极少数月末场景可能让同一交易日出现两行；npy 矩阵必须和 EOD 日期一一对应。
    out = out.drop_duplicates(subset=["交易日期"], keep="last")
    aligned = pd.DataFrame({"交易日期": pd.to_datetime(target_dates)})
    aligned = aligned.merge(out, on="交易日期", how="left")
    aligned.fillna(value=0, inplace=True)
    return aligned.rename(columns={"交易日期": "trade_date"})


def build_period_rebalance_matrix(dates: list[str], period_dict: dict[Any, list[Any]] | None = None) -> tuple[list[str], np.ndarray]:
    """把周期编号宽表转换成换仓日布尔矩阵。"""
    ids = build_period_id_frame(dates, period_dict=period_dict)
    period_names = [col for col in ids.columns if col != "trade_date"]
    values = ids[period_names].to_numpy(dtype=float)
    previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
    rebalance = (values != previous) & (values != 0)
    if rebalance.shape[0] > 0:
        rebalance[0, :] = False
    return period_names, rebalance.T.astype(bool, copy=False)


def save_period_files(eod_path: Path, period_path: Path) -> tuple[Path, Path, Path]:
    """从 ``eod/dates.npy`` 生成三个周期预运算文件。"""
    eod_path = Path(eod_path)
    period_path = Path(period_path)
    period_path.mkdir(parents=True, exist_ok=True)
    dates = [normalize_date_label(x) for x in decode_array(np.load(eod_path / "dates.npy", allow_pickle=True))]
    period_names, rebalance = build_period_rebalance_matrix(dates)

    names_path = period_path / "period.npy"
    mask_path = period_path / "period_dates.npy"
    dates_path = period_path / "dates.npy"
    np.save(names_path, np.asarray(period_names, dtype=object))
    np.save(mask_path, rebalance)
    np.save(dates_path, np.asarray(dates, dtype="S8"))
    return names_path, mask_path, dates_path
