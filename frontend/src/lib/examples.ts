/** Example tickets and limits for the analyze form. */
import type { TicketType } from '../api/types'

export const MAX_CHARS = 4000

export interface FormState {
  subject: string
  text: string
  ticket_type: TicketType | ''
  channel: string
  product: string
}

export const EMPTY_FORM: FormState = { subject: '', text: '', ticket_type: '', channel: '', product: '' }

export const EXAMPLES: { label: string; form: FormState }[] = [
  {
    label: 'Double charge',
    form: {
      subject: 'Charged twice',
      text: 'I was charged twice for my order this month and nobody has replied to my last two emails. Please refund the second charge.',
      ticket_type: 'Problem',
      channel: 'email',
      product: '',
    },
  },
  {
    label: 'Return question',
    form: {
      subject: 'Return window',
      text: 'How many days do I have to return headphones that I do not like?',
      ticket_type: 'Request',
      channel: 'chat',
      product: 'Headphones',
    },
  },
  {
    label: 'Outage',
    form: {
      subject: 'Service down',
      text: 'Our whole team cannot log in since this morning. The dashboard shows an error and this is urgent because we have a deadline today.',
      ticket_type: 'Incident',
      channel: 'web',
      product: '',
    },
  },
]
