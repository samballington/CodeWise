import { create } from 'zustand'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isError?: boolean
  isComplete?: boolean
}

export interface ContextActivity {
  id: string
  type: 'start' | 'search' | 'complete'
  message: string
  source?: string
  query?: string
  timestamp: Date
  sources?: string[]
  chunksFound?: number
  filesAnalyzed?: number
}

interface ChatStore {
  messages: Message[]
  addMessage: (message: Message) => void
  updateLastMessage: (updates: Partial<Message>) => void
  clearMessages: () => void
}

interface ContextStore {
  isGatheringContext: boolean
  currentActivity: ContextActivity | null
  recentActivities: ContextActivity[]
  setGatheringContext: (isGathering: boolean) => void
  addContextActivity: (activity: ContextActivity) => void
  clearContextActivities: () => void
}

interface ThemeStore {
  isDarkMode: boolean
  toggleTheme: () => void
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

export const useContextStore = create<ContextStore>((set) => ({
  isGatheringContext: false,
  currentActivity: null,
  recentActivities: [],
  
  setGatheringContext: (isGathering) =>
    set({ isGatheringContext: isGathering }),
  
  addContextActivity: (activity) =>
    set((state) => ({
      currentActivity: activity,
      recentActivities: [activity, ...state.recentActivities.slice(0, 9)], // Keep last 10
    })),
  
  clearContextActivities: () =>
    set({
      isGatheringContext: false,
      currentActivity: null,
      recentActivities: [],
    }),
}))

export const useThemeStore = create<ThemeStore>((set) => ({
  isDarkMode: true, // Default to dark mode
  toggleTheme: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
})) 