"""
2023分享会
author: 邢不行
微信: xbx9585
策略周期预运算
"""
import pandas as pd
import datetime


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


# 导入指数
def import_index_data(path, back_trader_start=None, back_trader_end=None):
    """
    从指定位置读入指数数据。指数数据来自于：program_back/构建自己的股票数据库/案例_获取股票最近日K线数据.py
    :param back_trader_end: 回测结束时间
    :param back_trader_start: 回测开始时间
    :param path:
    :return:
    """
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'], encoding='gbk')
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    df_index = df_index[['candle_end_time', '指数涨跌幅']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)

    if back_trader_start:
        df_index = df_index[df_index['交易日期'] >= pd.to_datetime(back_trader_start)]
    if back_trader_end:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(back_trader_end)]

    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


def calc_period_and_offset(period_dict, _index_data, end_day, min_day=2):
    """
    计算周期
    :param period_dict: 所有需要算的周期和offset，周期为key，offset放入list后组成value
    :param _index_data: 载入最新的指数数据
    :param end_day: 定义需要把这份数据产生到什么时候，不建议做太久远（以免出错）
    :param min_day: 定义单个周期的最小持仓周期，默认为2，即为最少持仓2天（没办法当天买当天卖）
    :return:
    """

    # ===通过TradeCalendar把index后的数据补一段
    lastday = _index_data['交易日期'].iloc[-1]
    # 离线造一份未来交易日数据
    day_win = max((pd.to_datetime(end_day) - lastday).days, 50)
    tc = TradeCalendar(day_win)
    calendar_data = tc.calendar[tc.calendar['trade_date'] > lastday].copy()
    if pd.to_datetime(end_day) > lastday + pd.to_timedelta('50d'):
        calendar_data = calendar_data[calendar_data['trade_date'] <= pd.to_datetime(end_day)]
    calendar_data.rename(columns={'trade_date': '交易日期'}, inplace=True)
    calendar_data = calendar_data[['交易日期']]
    # 把造出来的未来交易日数据拼接到原本的index交易日后面
    _index_data = pd.concat([_index_data, calendar_data], ignore_index=True)
    _index_data.sort_values(by=['交易日期'], ascending=True, ignore_index=True, inplace=True)
    # 增加'是否交易'和'周期最后交易日'列，准备划分周期
    _index_data['是否交易'] = 1
    _index_data['周期最后交易日'] = _index_data['交易日期']
    # 创造等待返回的大表
    all_period_offst_df = _index_data[['交易日期']].copy()

    agg_dict = {'周期最后交易日': 'last', '是否交易': 'sum'}

    for period_type in period_dict:  # 遍历周期
        if period_type in ['W53']:  # 周五买周三卖
            print(f'运算周期：{period_type} offset: 0')
            # 周五买周三卖的情况单独处理（周四还是会放在一个周期内，但是会变成负值）
            index_data = _index_data.copy()
            start_date = index_data['交易日期'].min()
            end_date = index_data['交易日期'].max()
            # ===第一步、先把W_3算一遍（周三尾盘卖，周四早盘买），目的时把周三定义出来
            index_data['交易日期'] -= pd.to_timedelta(f'3D')
            # 在df的最后一行添加一个交易日期为1990-01-01的行（此举的目的是统一resample的开始时间，有时候这行会多余）
            index_data.loc[index_data.index.max() + 1, '交易日期'] = pd.to_datetime("1990-01-01")
            # 将交易日期设置为index
            index_data.set_index('交易日期', inplace=True)
            period_df = index_data.resample(rule='W').agg(agg_dict)
            period_df.rename(columns={'是否交易': '交易天数'}, inplace=True)
            period_df = period_df[period_df['交易天数'] > 0]
            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)
            period_df[f'W53'] = 1
            # 至此period_df为按每周三分割的最后一个交易日

            # 把周四的信息拼接回交易日历中
            df = pd.merge(_index_data[['交易日期']].copy(), right=period_df[['交易日期', 'W53']], on='交易日期',
                          how='left')
            # 分割出每个周期
            df['W53_0'] = df['W53'].expanding().sum().shift()

            # ===第二步、把所有的周四标记出来,在原有的df里标记为None
            date_range_df = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='D'),
                                         columns=['交易日期'])
            date_range_df['周期最后交易日'] = date_range_df['交易日期'].copy()
            # 需要把周四都找出来
            date_range_df['交易日期'] -= pd.to_timedelta(f'4D')
            # 在df的最后一行添加一个交易日期为1990-01-01的行（此举的目的是统一resample的开始时间，有时候这行会多余）
            date_range_df.loc[date_range_df.index.max() + 1, '交易日期'] = pd.to_datetime("1990-01-01")
            # 将交易日期设置为index
            date_range_df.set_index('交易日期', inplace=True)
            period_df = date_range_df.resample(rule='W').agg({'周期最后交易日': 'last'})
            # 以上步骤找出周四后，把所有交易日中的周四定义为None
            df.loc[df['交易日期'].isin(period_df['周期最后交易日']), 'W53_0'] = None

            # ===第三步、查看所有分组中，是否有单交易日的情况（不满足T+1卖出的模式），举例2023-4-28
            counts = df['W53_0'].value_counts()
            # 找到只出现一次的数
            single_occurrence_values = counts[counts == 1].index
            # 将只出现一次的数赋值为None
            df.loc[df['W53_0'].isin(single_occurrence_values), 'W53_0'] = None

            # ===第四步、把所有None做ffill后变为负值。绝对值后的周期为一个名义周期（大概率为周五至周四），而实际周期为正值的周期（大概率为周五至周三）
            # 而在回测代码中，做groupby前会先把负的周期的涨跌幅定义为0，然后再按照名义周期进行groupby，这样累积涨跌幅其实时实际周期的值。
            df['_W53'] = df['W53_0'].copy()
            df['W53_0'].fillna(method='ffill', inplace=True)
            df.loc[pd.isnull(df['_W53']), 'W53_0'] = -df['W53_0']
            # 并入大表
            all_period_offst_df = pd.merge(left=all_period_offst_df, right=df[['交易日期', 'W53_0']],
                                           on='交易日期', how='left')
            continue  # 因为W53只有offset 0，所以这里直接可以进入下一个循环

        for offset in period_dict[period_type]:
            print(f'运算周期：{period_type} offset: {offset}')
            index_data = _index_data.copy()
            if type(period_type) == int:
                # period按天计算
                index_data['group'] = pd.Series((index_data.index - offset) / period_type).apply(int)
                period_df = index_data.groupby('group').agg(agg_dict)
                # 计算必须额外数据
                period_df['交易天数'] = period_type
            else:
                # period按 W 和 M的模式

                if (period_type == 'M') and (offset == -5):  # M -5需要单独算

                    # 这里单独处理月频周五的问题
                    start_date = index_data['交易日期'].min()
                    end_date = index_data['交易日期'].max()
                    # 做一列日频数据把非交易日期补齐
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    index_data = index_data.set_index('交易日期').reindex(date_range).reset_index()
                    index_data.rename(columns={'index': '交易日期'}, inplace=True)
                    index_data['周期最后交易日'].fillna(method='ffill', inplace=True)
                    index_data['是否交易'].fillna(value=0, inplace=True)
                    # 此时的index_data不再只有交易日，而是自然日，并且有是否交易的标志。

                    # 标记出自然月末最后一天
                    date_range_M = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='M'),
                                                columns=['交易日期'])
                    date_range_M['月末'] = 1
                    index_data = pd.merge(left=index_data, right=date_range_M, on='交易日期', how='left')

                    # 对于月末最后一天是交易日 且 下个月第一天是非交易日的，用月末最后一个交易日作为周期末，否则月末最后一个非交易日作为周期末
                    index_data.loc[(index_data['月末'] == 1) & (index_data['是否交易'].shift(-1) == 0), '是否交易'] = 0
                    index_data = index_data[index_data['是否交易'] == 0]
                    index_data.set_index('交易日期', inplace=True)
                    # 周期末对应的“周期最后交易日"字段，即为所求日期（周期末的卖出日）
                    period_df = index_data.resample(rule=period_type).agg(agg_dict)
                    period_df['交易天数'] = 20  # 这里如果要算对，还得重新再merge后再group，所以这里直接赋值20，目的仅通过后面的过滤
                else:
                    # 所有除W53外的W和M 0，都再这里计算
                    if (lambda s: any(char.isdigit() for char in s))(period_type):
                        if isinstance(offset, str) and 'D' in offset.upper():
                            # 231001 沈博文添加
                            offset = offset.upper()
                            index_data['交易日期'] -= pd.to_timedelta(offset)
                        else:
                            # nW的模式，均为周一换仓
                            index_data['交易日期'] -= pd.to_timedelta(f'{offset * 7}D')
                    else:
                        # W 的offset 0-4代表周一到周五的周期，因为M只有offset 0，所以等于没动
                        index_data['交易日期'] -= pd.to_timedelta(f'{offset}D')

                    # 在df的最后一行添加一个交易日期为1990-01-01的行（此举的目的是统一resample的开始时间）
                    index_data.loc[index_data.index.max() + 1, '交易日期'] = pd.to_datetime("1990-01-01")
                    # 将交易日期设置为index
                    index_data.set_index('交易日期', inplace=True)

                    period_df = index_data.resample(rule=period_type).agg(agg_dict)
                    period_df.rename(columns={'是否交易': '交易天数'}, inplace=True)

            period_df = period_df[period_df['交易天数'] > 0]  # 为了通过这里，所以M -5做了特殊处理人为定义交易天数为20

            # 算完一个周期offset后，需要再确认一下是否有单周期内低于min_day的情况（W53单独处理过了，不会在这里算）
            # 处理本周期只有1天的情况（A股当前T+1模式没办法当天卖），这里的处理方式为把当前周期并入下一个周期。
            # 举例：2021-10-08的W_0按理说是自成一个交易周期的，处理成:2021-09-30卖出持仓后，2021-10-08买入新持仓后不卖出直至2021-10-15.
            # 如果想人为定义持股必须3个交易日，可以直接在min_day传参为3.
            index_to_remove = []
            add_num = 0
            for index, row in period_df.iterrows():
                period_df.at[index, '交易天数'] += add_num
                add_num = 0
                if row['交易天数'] < min_day:
                    index_to_remove.append(index)
                    add_num = row['交易天数']
            period_df = period_df.drop(index_to_remove)

            period_df.reset_index(drop=True, inplace=True)
            period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)
            period_df[f'{period_type}_{offset}'] = 1
            # 并入大表
            all_period_offst_df = pd.merge(left=all_period_offst_df,
                                           right=period_df[['交易日期', f'{period_type}_{offset}']],
                                           on='交易日期', how='left')
            all_period_offst_df[f'{period_type}_{offset}'] = all_period_offst_df[
                f'{period_type}_{offset}'].expanding().sum().shift()

    all_period_offst_df.fillna(value=0, inplace=True)  # 2005年头部几日没算入第一个周期，所以处理一下空置
    return all_period_offst_df


if __name__ == '__main__':
    # 定义指数文件路径
    index_path = './index/sh000300.csv'
    # 保存周期文件的路径
    save_file = '../../data/period_offset.csv'
    # 周期计算的最大日期
    end_day = '2024-02-07'
    # 需要计算的周期offset合集key为周期，value为list格式包含周期对应的所有需要offset
    # 其中M -5 和W53为人为约定的特定周期
    period_dict = {
        2: [0, 1],
        3: [0, 1, 2],
        4: [0, 1, 2, 3],
        5: [0, 1, 2, 3, 4],
        10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'W': [0, 1, 2, 3, 4],
        '2W': [0, 1,'0D','1D','2D','3D','4D','7D','8D','9D','10D','11D'],
        '3W': [0, 1, 2],
        '4W': [0, 1, 2, 3],
        '5W': [0, 1, 2, 3, 4],
        '6W': [0, 1, 2, 3, 4, 5],
        'M': [0, -5],
        'W53': [0]
    }
    index_data = import_index_data(index_path)
    df = calc_period_and_offset(period_dict, index_data, end_day)
    pd.DataFrame(columns=['数据由邢不行整理，对数据字段有疑问的，可以直接微信私信邢不行，微信号：xbx9585']).to_csv(
        save_file,
        encoding='gbk',
        index=False)
    df.to_csv(save_file, encoding='gbk', index=False, mode='a')
