import { create } from 'zustand'

interface AppState {
  isSettingsOpen: boolean
  setSettingsOpen: (open: boolean) => void
  
  // Trigger to clear chat
  clearChatTrigger: number
  triggerClearChat: () => void
}

export const useAppStore = create<AppState>((set) => ({
  isSettingsOpen: false,
  setSettingsOpen: (open) => set({ isSettingsOpen: open }),
  
  clearChatTrigger: 0,
  triggerClearChat: () => set((state) => ({ clearChatTrigger: state.clearChatTrigger + 1 })),
}))
