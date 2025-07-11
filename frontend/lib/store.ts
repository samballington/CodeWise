import { create } from 'zustand'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isError?: boolean
  isComplete?: boolean
}

interface ChatStore {
  messages: Message[]
  addMessage: (message: Message) => void
  updateLastMessage: (updates: Partial<Message>) => void
  clearMessages: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  
  updateLastMessage: (updates) =>
    set((state) => {
      if (state.messages.length === 0) return state
      
      const messages = [...state.messages]
      const lastMessage = messages[messages.length - 1]
      messages[messages.length - 1] = { ...lastMessage, ...updates }
      
      return { messages }
    }),
  
  clearMessages: () => set({ messages: [] }),
})) 