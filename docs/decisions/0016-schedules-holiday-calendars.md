# 0016. Schedules and holiday calendars, split out of #17
Status: Accepted (2026-10-16, PR #47, issue #32)

## Context
Structured products (#31: accumulators, swaps, Asian caps) fix on business-day schedules and pay on separate dates. Products had only `expiry_date` and `ttm()`. #32 depended on #17 phase 1 (`HolidayCalendar`, `DayCount`), a larger market-data redesign.

## Decision
Ship only the #17 pieces #32 needs, in `quantlib_core`:
- `HolidayCalendar: Send + Sync` (`calendar.rs`): `is_business_day`, `business_days(start, end)`, `advance(d, n)`; `WeekendsOnly` and `HolidayList` (sorted, deduplicated, binary search).
- `BusinessDayConvention::{Following, ModifiedFollowing, Preceding}::adjust`.
- `Schedule { fixings, payments, pay_index }` (`schedule.rs`), private fields, validated in `new`: non-empty, one `pay_index` per fixing, fixings and payments strictly increasing, each payment on or after its fixing. `Schedule::monthly` groups fixings by calendar month, paid `lag` business days after the adjusted month end.
- `Schedule::times` converts with the existing `MarketCalendar`; dates on or before valuation map to t = 0.

## Alternatives considered
- Implement #17 phase 1 in full (`DayCount` enum, products off `Arc<dyn MarketCalendar>`): stays in #17; it changes every product.
- `times(valuation, day_count)` with a new `DayCount` (#32's signature): `DayCount` stays in #17; `MarketCalendar` is reused.

## Consequences
- Mid-life schedules (past fixings) are #40.
- Tests: `conventions_roll_weekends_and_month_ends`, `monthly_assigns_fixings_to_months_and_pays_after_lag`, `rejects_bad_schedules` in `quantlib_core/src/schedule.rs`; 69 black-box checks against a brute-force oracle were run during review, not committed.
