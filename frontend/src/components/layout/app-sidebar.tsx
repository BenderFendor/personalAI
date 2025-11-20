"use client"

import { Calendar, Home, Inbox, Search, Settings, ChevronRight, MessageSquare, Plus } from "lucide-react"
import { useAppStore } from "@/lib/store"
import { ModeToggle } from "@/components/mode-toggle"
import { SettingsDialog } from "@/components/settings/settings-dialog"
import { SearchDialog } from "@/components/search/search-dialog"
import { useEffect, useState } from "react"
import { getSessions, loadSession, Session, clearHistory, newChat } from "@/lib/api"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarHeader,
} from "@/components/ui/sidebar"

export function AppSidebar() {
  const { setSettingsOpen, triggerClearChat, isSettingsOpen, triggerReloadChat, currentSessionId, setCurrentSessionId, setMessages } = useAppStore()
  const [sessions, setSessions] = useState<Session[]>([])
  const [isSearchOpen, setSearchOpen] = useState(false)

  const refreshSessions = async () => {
    try {
      const data = await getSessions()
      setSessions(data)
    } catch (error) {
      console.error("Failed to fetch sessions", error)
    }
  }

  useEffect(() => {
    refreshSessions()
  }, [isSettingsOpen, currentSessionId])

  const handleNewChat = async (e: React.MouseEvent) => {
    e.preventDefault()
    try {
      const data = await newChat()
      setCurrentSessionId(data.session_id)
      setMessages([])
      // Force a refresh of the session list to show the new one
      await refreshSessions()
    } catch (error) {
      console.error("Failed to create new chat", error)
    }
  }

  const handleSettings = (e: React.MouseEvent) => {
    e.preventDefault()
    setSettingsOpen(true)
  }

  const handleSearch = (e: React.MouseEvent) => {
    e.preventDefault()
    setSearchOpen(true)
  }

  const handleLoadSession = async (sessionId: string) => {
    try {
      const data = await loadSession(sessionId)
      setCurrentSessionId(sessionId)
      // Update messages in store
      if (data.messages) {
        setMessages(data.messages)
      } else {
        // Fallback to fetching history if loadSession doesn't return messages
        triggerReloadChat()
      }
    } catch (error) {
      console.error("Failed to load session", error)
    }
  }

  return (
    <>
      <Sidebar className="border-r border-border/50 bg-sidebar/50 backdrop-blur-xl">
        <SidebarHeader className="p-4 border-b border-border/50">
          <div className="flex items-center gap-2 px-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center text-primary-foreground font-bold shadow-lg">
              AI
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-sm">Personal AI</span>
              <span className="text-[10px] text-muted-foreground">Local Intelligence</span>
            </div>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupContent className="px-2 py-2">
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton 
                    onClick={handleNewChat}
                    className="bg-primary/10 text-primary hover:bg-primary/20 hover:text-primary transition-colors justify-center border border-primary/20 shadow-sm mb-4 h-10"
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    <span className="font-medium">New Chat</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                
                <Collapsible defaultOpen className="group/collapsible">
                  <SidebarMenuItem>
                    <CollapsibleTrigger asChild>
                      <SidebarMenuButton className="font-medium text-muted-foreground hover:text-foreground">
                        <MessageSquare className="h-4 w-4" />
                        <span>Recent Chats</span>
                        <ChevronRight className="ml-auto h-4 w-4 transition-transform group-data-[state=open]/collapsible:rotate-90 opacity-50" />
                      </SidebarMenuButton>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <SidebarMenuSub>
                        {sessions.length === 0 && (
                          <div className="px-4 py-2 text-xs text-muted-foreground/50 italic">
                            No history yet
                          </div>
                        )}
                        {sessions.slice(0, 10).map((session) => (
                          <SidebarMenuSubItem key={session.id}>
                            <SidebarMenuSubButton 
                              onClick={() => handleLoadSession(session.id)}
                              isActive={currentSessionId === session.id}
                              className="cursor-pointer transition-all duration-200"
                            >
                              <span className="truncate text-xs">{session.title || "Untitled Chat"}</span>
                            </SidebarMenuSubButton>
                          </SidebarMenuSubItem>
                        ))}
                      </SidebarMenuSub>
                    </CollapsibleContent>
                  </SidebarMenuItem>
                </Collapsible>

                <SidebarGroupLabel className="mt-4">Tools</SidebarGroupLabel>
                <SidebarMenuItem>
                  <SidebarMenuButton onClick={handleSearch}>
                    <Search className="h-4 w-4" />
                    <span>RAG Search</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                  <SidebarMenuButton onClick={handleSettings}>
                    <Settings className="h-4 w-4" />
                    <span>Settings</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter className="p-4 border-t border-border/50 bg-sidebar/50">
          <div className="flex items-center justify-between bg-muted/50 p-2 rounded-lg border border-border/50">
            <span className="text-xs font-medium text-muted-foreground">Dark Mode</span>
            <ModeToggle />
          </div>
        </SidebarFooter>
      </Sidebar>
      <SettingsDialog open={isSettingsOpen} onOpenChange={setSettingsOpen} />
      <SearchDialog open={isSearchOpen} onOpenChange={setSearchOpen} />
    </>
  )
}
