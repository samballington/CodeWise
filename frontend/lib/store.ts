import { create } from 'zustand'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  output?: string  // New field for canonical response contract
  timestamp: Date
  isError?: boolean
  isComplete?: boolean
  isProcessing?: boolean
  structuredResponse?: any
  contextData?: {
    sources?: string[]
    chunksFound?: number
    filesAnalyzed?: number
    query?: string
  }
  toolCalls?: Array<{
    tool: string
    input: string
    output: string
    status: 'running' | 'completed' | 'error'
  }>
  bufferedToolCalls?: Array<{
    tool: string
    input: string
    output: string
    status: 'running' | 'completed' | 'error'
  }>
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
  addToolCallToLastMessage: (toolCall: { tool: string; input: string; output: string; status: 'running' | 'completed' | 'error' }) => void
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

interface ModelStore {
  selectedModel: string
  setSelectedModel: (model: string) => void
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
  
  addToolCallToLastMessage: (toolCall) =>
    set((state) => {
      if (state.messages.length === 0) return state
      
      const messages = [...state.messages]
      const lastMessage = messages[messages.length - 1]
      const existingToolCalls = lastMessage.toolCalls || []
      
      messages[messages.length - 1] = {
        ...lastMessage,
        toolCalls: [...existingToolCalls, toolCall]
      }
      
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

export const useModelStore = create<ModelStore>((set) => ({
  selectedModel: typeof window !== 'undefined' 
    ? localStorage.getItem('codewise-selected-model') || 'gpt-oss-120b'
    : 'gpt-oss-120b',
  
  setSelectedModel: (model) => {
    set({ selectedModel: model })
    if (typeof window !== 'undefined') {
      localStorage.setItem('codewise-selected-model', model)
    }
  },
})) 