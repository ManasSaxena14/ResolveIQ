import type { TicketType } from '../api/types'

export interface Ticket {
  id: string
  subject: string
  text: string
  ticket_type: TicketType | ''
  channel: string
  product: string
  received: string
}

/** The ten queues the classifier sorts into, in the order they stand on the rack. */
export const QUEUES = [
  'Billing and Payments',
  'Returns and Exchanges',
  'Customer Service',
  'General Inquiry',
  'Sales and Pre-Sales',
  'Product Support',
  'Technical Support',
  'IT Support',
  'Service Outages and Maintenance',
  'Human Resources',
] as const

/** Invented tickets for the fictional retailer Altura, so the app opens on real work. */
export const SAMPLE_TICKETS: Ticket[] = [
  {
    id: 'charged-twice',
    subject: 'Charged twice for one order',
    text: 'I was charged twice for my order this month and nobody has replied to my last two emails. Please refund the second charge.',
    ticket_type: 'Problem',
    channel: 'email',
    product: '',
    received: '9:42',
  },
  {
    id: 'locked-out',
    subject: 'Whole team locked out',
    text: 'Our whole team has been locked out of the business account since this morning. The sign-in page shows a server error. This is urgent, we have a deadline today.',
    ticket_type: 'Incident',
    channel: 'web',
    product: '',
    received: '9:38',
  },
  {
    id: 'return-window',
    subject: 'Returning headphones',
    text: 'How many days do I have to return headphones that I do not like? The box is open but everything is inside.',
    ticket_type: 'Request',
    channel: 'chat',
    product: 'Headphones',
    received: '9:31',
  },
  {
    id: 'missing-package',
    subject: 'Marked delivered, nothing here',
    text: 'Tracking says my package was delivered yesterday, but there is nothing at my door and the neighbours do not have it either. What happens now?',
    ticket_type: 'Problem',
    channel: 'email',
    product: '',
    received: '9:24',
  },
  {
    id: 'broken-speaker',
    subject: 'Speaker arrived broken',
    text: 'My new speaker arrived in a dented box and does not turn on. I need it for Saturday. Can you send a replacement?',
    ticket_type: 'Incident',
    channel: 'chat',
    product: 'Speaker',
    received: '9:17',
  },
  {
    id: 'password-changed',
    subject: 'Password changed, not by me',
    text: 'I got an email saying my password was changed, but I did not change it. I think someone is in my account and my card is saved there. Please lock it now.',
    ticket_type: 'Incident',
    channel: 'email',
    product: '',
    received: '9:05',
  },
  {
    id: 'cancel-plus',
    subject: 'Cancel Altura Plus',
    text: 'Please cancel my Altura Plus membership before it renews next week. Will I get any money back for the months I have left?',
    ticket_type: 'Request',
    channel: 'email',
    product: 'Altura Plus',
    received: '8:56',
  },
  {
    id: 'codes-missing',
    subject: 'Sign-in codes never arrive',
    text: 'The two-factor codes never reach my phone, so I cannot sign in. I have tried three times and restarted the phone.',
    ticket_type: 'Problem',
    channel: 'chat',
    product: '',
    received: '8:44',
  },
  {
    id: 'cracked-screen',
    subject: 'Is a cracked screen covered?',
    text: 'Does the warranty cover a cracked screen after I dropped my phone, or do I need a protection plan for that?',
    ticket_type: 'Request',
    channel: 'web',
    product: 'Phone',
    received: '8:31',
  },
  {
    id: 'trade-in',
    subject: 'Trade-in value went down',
    text: 'I was quoted one value for my old tablet and received less after you inspected it. Nobody explained why. I want the original amount or the tablet back.',
    ticket_type: 'Problem',
    channel: 'phone',
    product: 'Tablet',
    received: '8:20',
  },
  {
    id: 'express-shipping',
    subject: 'Express to Norway',
    text: 'Do you ship to Norway, and how long does express take? I would also like to know who pays the import fees.',
    ticket_type: 'Request',
    channel: 'web',
    product: '',
    received: '8:12',
  },
  {
    id: 'careers',
    subject: 'Engineering internships',
    text: 'Does Altura offer internships for software engineers next summer, and where do I apply?',
    ticket_type: 'Request',
    channel: 'email',
    product: '',
    received: '8:03',
  },
]

export const MAX_CHARS = 4000

/** "email, problem, received 9:42" with whatever is known. */
export function ticketMeta(ticket: Ticket): string {
  const parts = [ticket.channel, ticket.ticket_type.toLowerCase(), ticket.product].filter(Boolean)
  const line = [...parts, ticket.received && `received ${ticket.received}`].filter(Boolean).join(', ')
  return line.charAt(0).toUpperCase() + line.slice(1)
}

/** What the composer holds before it becomes a ticket. */
export interface ComposerDraft {
  subject: string
  text: string
  ticket_type: TicketType | ''
  channel: string
  product: string
}

export const EMPTY_DRAFT: ComposerDraft = { subject: '', text: '', ticket_type: '', channel: '', product: '' }
