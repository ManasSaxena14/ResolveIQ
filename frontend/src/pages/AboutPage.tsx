import { SortingRack } from '../components/SortingRack'
import type { Category } from '../api/types'

const REPO = 'https://github.com/ManasSaxena14/ResolveIQ'

/** The classifier's real output for the first sample ticket. */
const EXAMPLE: Category = {
  label: 'Billing and Payments',
  confidence: 0.9922,
  needs_review: false,
  top: [
    { label: 'Billing and Payments', probability: 0.9922 },
    { label: 'Returns and Exchanges', probability: 0.0027 },
    { label: 'Human Resources', probability: 0.002 },
    { label: 'Technical Support', probability: 0.0009 },
    { label: 'Product Support', probability: 0.0006 },
    { label: 'Customer Service', probability: 0.0005 },
    { label: 'IT Support', probability: 0.0003 },
    { label: 'Sales and Pre-Sales', probability: 0.0003 },
    { label: 'Service Outages and Maintenance', probability: 0.0003 },
    { label: 'General Inquiry', probability: 0.0001 },
  ],
}

const RESULTS = [
  {
    part: 'Choosing the queue',
    measure: 'Macro F1, ten queues',
    baseline: '0.633, TF-IDF',
    ours: '0.521, DistilBERT',
    note: 'It names the right queue for 47% of tickets and is still behind the simple method.',
  },
  {
    part: 'Scoring the risk',
    measure: 'PR-AUC',
    baseline: '0.744, logistic regression',
    ours: '0.749, XGBoost',
    note: 'The flag catches 87% of high-priority tickets and is raised on 60% of all tickets.',
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
            <p>A fine-tuned DistilBERT model weighs all ten queues. When it is unsure, it asks for a person.</p>
          </li>
          <li>
            <h3>The risk is scored</h3>
            <p>A calibrated XGBoost model estimates how likely the ticket is to escalate and names its reasons.</p>
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
          Measured once on tickets the models had never seen, next to the simplest method that could work.
        </p>
        <div className="frame">
          <div className="table-scroll">
            <table className="stacked">
              <thead>
                <tr>
                  <th scope="col">Part</th>
                  <th scope="col">Measure</th>
                  <th scope="col">Simple method</th>
                  <th scope="col">ResolveIQ</th>
                  <th scope="col">In plain words</th>
                </tr>
              </thead>
              <tbody>
                {RESULTS.map((r) => (
                  <tr key={r.part}>
                    <td className="strong">{r.part}</td>
                    <td data-label="Measure">{r.measure}</td>
                    <td data-label="Simple method">{r.baseline}</td>
                    <td data-label="ResolveIQ">{r.ours}</td>
                    <td>{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <p>
          <a href={`${REPO}/blob/main/docs/evaluation_report.md`}>Read the full evaluation</a>
        </p>
      </section>

      <section>
        <h2>What it cannot do</h2>
        <ul className="limits">
          <li>The tickets and help articles are invented, so these numbers do not predict a real company's results.</li>
          <li>The data has no record of real escalations. High priority stands in for them.</li>
          <li>
            The models react to wording more than meaning. A lockout that sounds calm can score low, and a polite question
            can score high.
          </li>
          <li>Reply quality has not been measured with a language model yet.</li>
          <li>It reads English only, and it never sends anything on its own.</li>
        </ul>
      </section>
    </div>
  )
}
