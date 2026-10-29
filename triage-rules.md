# SpiceJet Triage Rules and Scoring Configuration

This document specifies the deterministic triage scoring rules, safety overrides, prompt-injection patterns, and classification definitions used by the AI Social Listening and Crisis Response system.

```yaml
spread_alert_threshold: 0.40
min_confidence: 0.60
sentiment_points:
  Negative: 2
  Neutral: 0
  Positive: 0
urgency_points:
  High: 3
  Medium: 1
  Low: 0
spread_points:
  high: 3       # spread_proba >= spread_alert_threshold
  moderate: 1   # spread_proba >= 0.6 * spread_alert_threshold
  low: 0        # spread_proba < 0.6 * spread_alert_threshold
virality_points:
  Viral: 3
  High: 2
  Moderate: 1
  Low: 0
virality_tiers:
  Moderate: 30
  High: 60
  Viral: 80
holdout_fraction: 0.25
model_selection_metric: f2
target_recall: 0.75
risk_levels:
  Critical: 8
  High: 6
  Medium: 3
  Low: 0
safety_override_level: Critical
safety_keywords:
  - injury
  - injured
  - hurt
  - medical emergency
  - ambulance
  - smoke
  - fire
  - engine
  - security threat
  - harassment
  - assault
  - unsafe
  - emergency landing
  - oxygen mask
  - crash
  - accident
injection_patterns:
  - "ignore previous instructions"
  - "disregard all instructions"
  - "system prompt"
  - "bypass"
  - "output the word"
  - "classify this as positive"
  - "label this as"
  - "new instruction:"
  - "print your prompt"
  - "reveal your instructions"
  - "administrative override"
platform_reply_limits:
  X: 280
  Instagram: 500
  YouTube: 500
  Reddit: 800
theme_window_hours: 72
theme_growth_ratio: 1.5
theme_min_posts: 5
```

## Urgency definitions
- **High**: The passenger is in an acute, ongoing crisis requiring immediate intervention. Examples include: currently stranded at the airport due to sudden cancellation or major delay without information, missed connection in progress, lost luggage containing vital medication, active medical issue, safety concern, or severe distress where minutes matter.
- **Medium**: Significant operational friction or inconvenience that has already occurred or requires timely resolution within a few hours. Examples include: unnotified flight delays, baggage delayed at the carousel, pending refund requests with no status updates, unanswered customer service complaints, or rebooking questions during travel day.
- **Low**: General questions, routine information inquiries, policy clarifications, compliments, praise, or past experiences with no pending operational impact. Examples include: baggage allowance queries, web check-in questions days before travel, positive reviews, or general brand feedback.

## Topic definitions
- **Flight Delay**: Mentions specifically referencing schedule delays, late departures, late arrivals, waiting at the gate, missed connections due to late flights, or shifting departure times.
- **Flight Cancellation**: Mentions regarding flights that have been called off or cancelled, last-minute cancellation notifications, stranded passengers needing rebooking due to cancelled sectors, or queueing for cancellation assistance.
- **Baggage Issue**: Mentions concerning delayed luggage, baggage belt stoppages, lost bags, broken or damaged suitcases, missing baggage tags, or delivery of baggage.
- **Refund**: Mentions related to ticket refund status, delayed refund payments, cancellation fee charges, billing discrepancies, duplicate card charges, or refund timeline disputes.
- **Booking/Check-in**: Mentions dealing with web check-in errors, seat allocation, name corrections, booking confirmations, website loading failures, or boarding pass download issues.
- **Customer Service**: Mentions describing interactions with airline staff, ground attendants, support call centers, unresponsive complaints, customer care attitude, or crew assistance.
- **General Enquiry**: General questions about airline policies, flight schedules, route queries, boarding cut-off times, baggage size allowances, or special assistance availability.
- **Positive Experience**: Praise, gratitude, compliments on flight crew, smooth landings, pleasant flights, helpful ground staff, or positive travel reviews.

## Sentiment definitions
- **Positive**: Expresses satisfaction, gratitude, compliments, relief, or a constructive appreciation for airline services or crew.
- **Neutral**: Purely informational inquiries, objective questions, neutral statements of fact, or requests for policy details without positive or negative emotional charge.
- **Negative**: Expresses frustration, anger, disappointment, sarcasm, distress, criticism, or complaints regarding flight disruptions, lost items, delays, or poor service. Sarcastic praise about a failure is always categorized as Negative.
