# Example explanations

The five test tickets with the highest and the three with the lowest calibrated
escalation probability, with the reasons returned by the model. Subjects come from
the synthetic public dataset and are already masked.

**The company's data analytics system unexpectedly experienced a failure. Recent updates to **

- Probability 0.96, alert: yes, actual priority: high
- Ticket type Incident, queue Technical Support
- Reasons: Wording typical of high-priority tickets: "system", "failure", "the platform"; Predicted queue is Technical Support (60%)

**Dear Customer Support, our marketing agency is facing system outages impacting our digital**

- Probability 0.96, alert: yes, actual priority: high
- Ticket type Incident, queue IT Support
- Reasons: Wording typical of high-priority tickets: "outages", "system outages", "facing system"; Low chance of the Service Outages and Maintenance queue (41%)

**To Customer Support, reporting a significant performance decline noticed during peak activ**

- Probability 0.95, alert: yes, actual priority: high
- Ticket type Incident, queue Technical Support
- Reasons: Wording typical of high-priority tickets: "server capacity", "activity", "capacity"; Low chance of the Technical Support queue (44%); Low chance of the Product Support queue (10%)

**Concern Regarding Data Analytics System**

- Probability 0.95, alert: yes, actual priority: high
- Ticket type Incident, queue Service Outages and Maintenance
- Reasons: Wording typical of high-priority tickets: "system", "unexpected outage", "outage"; Low chance of the Billing and Payments queue (0%); Predicted queue is Service Outages and Maintenance (77%)

**encountered a data breach in hospital systems, resulting in the compromise of patient reco**

- Probability 0.95, alert: yes, actual priority: high
- Ticket type Problem, queue Technical Support
- Reasons: Wording typical of high-priority tickets: "urgent", "firewall and", "systems resulting"

**Variability in Digital Strategy Metrics Among Various Platforms**

- Probability 0.02, alert: no, actual priority: low
- Ticket type Incident, queue Product Support
- Reasons: No feature raises the score

**Seeking Information on Zoho CRM Integration Capabilities**

- Probability 0.02, alert: no, actual priority: low
- Ticket type Request, queue Human Resources
- Reasons: No feature raises the score

**Problem with Delayed Payroll Processing**

- Probability 0.02, alert: no, actual priority: medium
- Ticket type Incident, queue Human Resources
- Reasons: Negative tone (sentiment -0.68)
