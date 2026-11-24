import { SortingRack } from '../components/SortingRack'
import type { Category } from '../api/types'

const REPO = 'https://github.com/ManasSaxena14/ResolveIQ'

/** The classifier's real output for the first sample ticket. */
const EXAMPLE: Category = {
  label: 'Billing and Payments',
  confidence: 0.9862,
  needs_review: false,
  top: [
    { label: 'Billing and Payments', probability: 0.9862 },
    { label: 'Customer Service', probability: 0.0058 },
    { label: 'Returns and Exchanges', probability: 0.0056 },
    { label: 'Sales and Pre-Sales', probability: 0.0014 },
    { label: 'Technical Support', probability: 0.0006 },
    { label: 'Service Outages and Maintenance', probability: 0.0003 },
    { label: 'Human Resources', probability: 0.0001 },
    { label: 'General Inquiry', probability: 0 },
    { label: 'IT Support', probability: 0 },
    { label: 'Product Support', probability: 0 },
  ],
}

const RESULTS = [
  {
    part: 'Choosing the queue',
    measure: 'Right queue, 40 shop tickets',
    baseline: '17, DistilBERT',
    ours: '34',
    note: 'When it was sure enough to send a ticket on, it was right 14 out of 14 times.',
  },
  {
    part: 'Scoring the risk',
    measure: 'ROC-AUC, same 40 tickets',
    baseline: '0.527, XGBoost',
    ours: '0.922',
    note: 'It flags 5 of 8 urgent tickets and 3 of 32 calm ones. The earlier model flagged 36 of all 40.',
  },
  {
    part: 'Public dataset',
    measure: 'Queue accuracy, 2,451 tickets',
    baseline: '47%, DistilBERT',
    ours: '75%',
    note: 'Most of these nearly repeat a training ticket. On the least similar ones it is right 32% of the time.',
  },
  {
    part: 'Finding articles',
    measure: 'Right article found, held-out questions',
    baseline: 'Not measured',
    ours: '7 of 7',
    note: 'It did not refuse any of 3 questions the help center cannot answer.',
  },
]

export function AboutPage() {
  return (
    <div className="page about">
      <section className="about-hero">
        <div>
          <h1>It sorts support tickets and shows you why.</h1>
          <p className="lead">
            ResolveIQ reads a support ticket and does the three things an agent does first: finds the answer, picks the
            queue and judges whether it is about to escalate. It shows its reasons, and a person decides.
          </p>
          <div className="row">
            <a className="button primary" href="#/triage">
              Open the inbox
            </a>
            <a className="button" href={REPO}>
              Read the code
            </a>
          </div>
        </div>
        <SortingRack category={EXAMPLE} subject="Charged twice for one order" />
      </section>

      <section>
        <h2>What happens to a ticket</h2>
        <ol className="steps">
          <li>
            <h3>Personal details are masked</h3>
            <p>Emails, phone numbers and card numbers are replaced before the queue and risk models read the text.</p>
          </li>
          <li>
            <h3>A queue is chosen</h3>
            <p>Its meaning is compared with past tickets and with examples from the shop. When that is unsure, it asks for a person.</p>
          </li>
          <li>
            <h3>The risk is scored</h3>
            <p>Similar past tickets, urgent examples and an XGBoost model estimate how likely it is to escalate, with reasons.</p>
          </li>
          <li>
            <h3>A reply is written</h3>
            <p>
              Help articles are found by meaning and by keyword, and a language model writes from those alone, citing
              each one. With no article, it writes nothing.
            </p>
          </li>
        </ol>
      </section>

      <section>
        <h2>How well it works</h2>
        <p className="measure">
          Measured once on tickets the models had never seen, next to the models they replaced.
        </p>
        <div className="frame">
          <div className="table-scroll">
            <table className="stacked">
              <thead>
                <tr>
                  <th scope="col">Part</th>
                  <th scope="col">Measure</th>
                  <th scope="col">Earlier model</th>
                  <th scope="col">ResolveIQ</th>
                  <th scope="col">In plain words</th>
                </tr>
              </thead>
              <tbody>
                {RESULTS.map((r) => (
                  <tr key={r.part}>
                    <td className="strong">{r.part}</td>
                    <td data-label="Measure">{r.measure}</td>
                    <td data-label="Earlier model">{r.baseline}</td>
                    <td data-label="ResolveIQ">{r.ours}</td>
                    <td>{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p>
          <a href={`${REPO}/blob/main/docs/semantic_model.md`}>Read the full evaluation</a>
        </p>
      </section>

      <section>
        <h2>What it cannot do</h2>
        <ul className="limits">
          <li>The tickets and help articles are invented, so these numbers do not predict a real company's results.</li>
          <li>The data has no record of real escalations. High priority stands in for them.</li>
          <li>
            The shop tickets were written for this project and 40 is a small sample, so the true share of right queues
            could be anywhere from 71% to 93%.
          </li>
          <li>Urgent tickets in calm business language can be missed, so the order of the inbox is a help, not a promise.</li>
          <li>Reply quality has not been measured with a language model yet.</li>
          <li>It reads English only, and it never sends anything on its own.</li>
        </ul>
      </section>
    </div>
  )
}
