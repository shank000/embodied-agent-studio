"""Versioned national holiday dates for public token-price estimates.

Source: State Council General Office, 国办发明电〔2025〕7号, 2025-11-04.
Official government republication checked 2026-10-22. Only 2026 is covered.
The announced nationwide holiday/rest periods interpret DeepSeek's otherwise
undefined "Chinese public holidays"; regional/individual leave is not included.
This calendar never turns a make-up working Saturday/Sunday into a weekday.
"""
from datetime import date, datetime, timedelta, timezone

CHINA_TIME = timezone(timedelta(hours=8))
SOURCE_URL = 'https://www.beijing.gov.cn/zhengce/zhengcefagui/202511/t20251104_4258873.html'
PRIMARY_URL = 'https://www.gov.cn/zhengce/zhengceku/202511/content_7047091.htm'
CALENDAR_ID = 'cn-national-2026-gbfmd-2025-7-v1'
_HOLIDAY_PERIODS = (
    ('2026-01-01', '2026-01-03'), ('2026-02-15', '2026-02-23'),
    ('2026-04-04', '2026-04-06'), ('2026-05-01', '2026-05-05'),
    ('2026-06-19', '2026-06-21'), ('2026-09-25', '2026-09-27'),
    ('2026-10-01', '2026-10-07'),
)


def resolve_public_holiday(instant, region):
    """Return a sourced calendar decision, or unknown outside explicit coverage."""
    if not isinstance(instant, datetime) or instant.tzinfo is None:
        return {'known': False, 'message': '调用时间缺失或无时区，不能匹配节假日日历。'}
    local_day = instant.astimezone(CHINA_TIME).date()
    result = {'known': False, 'region': region, 'year': local_day.year, 'date': local_day.isoformat(),
              'timezone': 'UTC+08:00'}
    if region != 'CN' or local_day.year != 2026:
        result['message'] = f'尚未收录 {region} {local_day.year} 年经核对的官方节假日日历，保留峰谷范围。'
        return result
    holiday = any(date.fromisoformat(first) <= local_day <= date.fromisoformat(last) for first, last in _HOLIDAY_PERIODS)
    return {**result, 'known': True, 'is_holiday': holiday, 'calendar_id': CALENDAR_ID,
            'source_url': SOURCE_URL, 'primary_url': PRIMARY_URL, 'document_date': '2025-11-04',
            'checked_at': '2026-10-22', 'coverage': '2026',
            'interpretation': '按国务院年度全国放假调休安排解释 Chinese public holidays；不因周末补班改变周一至周五限制。'}
