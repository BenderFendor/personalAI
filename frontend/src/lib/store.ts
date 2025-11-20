import { create } from 'zustand'

export interface Message {
  role: "user" | "assistant" | "tool"
  content: string
  thinking?: string
  tool_calls?: any[]
  id?: string
}

export interface Session {
  id: string
  title: string
  file: string
  timestamp: string
}

interface AppState {
  isSettingsOpen: boolean
  setSettingsOpen: (open: boolean) => void
  
  // Chat State
  messages: Message[]
  setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void
  addMessage: (message: Message) => void
  updateLastMessage: (updater: (msg: Message) => Message) => void
  
  currentSessionId: string | null
  setCurrentSessionId: (id: string) => void
  
  sessions: Session[]
  setSessions: (sessions: Session[]) => void
  
  // Triggers (keeping for compatibility for now, but should phase out)
  clearChatTrigger: number
  triggerClearChat: () => void
  reloadChatTrigger: number
  triggerReloadChat: () => void
}

export const useAppStore = create<AppState>((set) => ({
  isSettingsOpen: false,
  setSettingsOpen: (open) => set({ isSettingsOpen: open }),
  
  messages: [],
  setMessages: (messages) => set((state) => ({ 
    messages: typeof messages === 'function' ? messages(state.messages) : messages 
  })),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  updateLastMessage: (updater) => set((state) => {
    const msgs = [...state.messages]
    if (msgs.length === 0) return state
    msgs[msgs.length - 1] = updater(msgs[msgs.length - 1])
    return { messages: msgs }
  }),
  
  currentSessionId: null,
  setCurrentSessionId: (id) => set({ currentSessionId: id }),
  
  sessions: [],
  setSessions: (sessions) => set({ sessions }),
  
  clearChatTrigger: 0,
  triggerClearChat: () => set((state) => ({ clearChatTrigger: state.clearChatTrigger + 1 })),
  
  reloadChatTrigger: 0,
  triggerReloadChat: () => set((state) => ({ reloadChatTrigger: state.reloadChatTrigger + 1 })),
}))
