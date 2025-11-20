"use client"

import React, { useState, useEffect } from 'react'
import { CabinetSidebar, CabinetId, CABINETS } from './cabinet-sidebar'
import { FolderStack } from './folder-stack'
import { useAppStore } from '@/lib/store'
import { getSessions, Session, newChat, loadSession } from '@/lib/api'
import { ChatInterface } from '@/components/chat/chat-interface'

export function TactileLayout() {
  const { currentSessionId, setCurrentSessionId, setMessages, triggerReloadChat, isSettingsOpen } = useAppStore()
  const [activeCabinet, setActiveCabinet] = useState<CabinetId>('today')
  const [sessions, setSessions] = useState<Session[]>([])
  const [filteredSessions, setFilteredSessions] = useState<Session[]>([])
  const [counts, setCounts] = useState<Record<CabinetId, number>>({
    today: 0, yesterday: 0, week: 0, older: 0, saved: 0
  })

  const refreshSessions = async () => {
    try {
      const data = await getSessions()
      setSessions(data)
      categorizeSessions(data)
    } catch (error) {
      console.error("Failed to fetch sessions", error)
    }
  }

  const categorizeSessions = (allSessions: Session[]) => {
    const now = new Date()
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
    const yesterday = today - 86400000
    const weekAgo = today - 86400000 * 7

    const newCounts = { today: 0, yesterday: 0, week: 0, older: 0, saved: 0 }
    
    allSessions.forEach(s => {
      const date = new Date(s.started_at || Date.now()).getTime()
      if (date >= today) newCounts.today++
      else if (date >= yesterday) newCounts.yesterday++
      else if (date >= weekAgo) newCounts.week++
      else newCounts.older++
    })
    
    setCounts(newCounts)
  }

  useEffect(() => {
    refreshSessions()
  }, [currentSessionId, isSettingsOpen])

  useEffect(() => {
    if (!sessions.length) return

    const now = new Date()
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
    const yesterday = today - 86400000
    const weekAgo = today - 86400000 * 7

    const filtered = sessions.filter(s => {
      const date = new Date(s.started_at || Date.now()).getTime()
      switch (activeCabinet) {
        case 'today': return date >= today
        case 'yesterday': return date >= yesterday && date < today
        case 'week': return date >= weekAgo && date < yesterday
        case 'older': return date < weekAgo
        default: return true
      }
    })
    
    // Sort by date desc
    filtered.sort((a, b) => new Date(b.started_at || 0).getTime() - new Date(a.started_at || 0).getTime())
    
    setFilteredSessions(filtered)
  }, [activeCabinet, sessions])

  const handleNewChat = async () => {
    try {
      const data = await newChat()
      setCurrentSessionId(data.session_id)
      setMessages([])
      await refreshSessions()
      setActiveCabinet('today') // Switch to today for new chat
    } catch (error) {
      console.error("Failed to create new chat", error)
    }
  }

  const handleSelectSession = async (sessionId: string) => {
    try {
      const data = await loadSession(sessionId)
      setCurrentSessionId(sessionId)
      if (data.messages) {
        setMessages(data.messages)
      } else {
        triggerReloadChat()
      }
    } catch (error) {
      console.error("Failed to load session", error)
    }
  }

  const activeCabinetName = CABINETS.find(c => c.id === activeCabinet)?.name || 'Unknown Cabinet'

  return (
    <div className="flex h-screen w-full bg-[#121212] text-stone-200 font-sans overflow-hidden">
      <CabinetSidebar 
        activeCabinet={activeCabinet} 
        setActiveCabinet={setActiveCabinet} 
        counts={counts}
      />
      
      <FolderStack 
        sessions={filteredSessions}
        selectedSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        activeCabinetName={activeCabinetName}
      />

      <main className="flex-1 h-full relative bg-[#121212] flex flex-col">
        {currentSessionId ? (
           <ChatInterface />
         ) : (
           <div className="flex items-center justify-center h-full text-stone-700">
             <div className="text-center">
               <div className="w-16 h-16 border-2 border-stone-800 rounded-full mx-auto mb-4 border-dashed animate-spin-slow"></div>
               <p className="font-mono text-xs uppercase tracking-widest">System Ready</p>
             </div>
           </div>
         )}
      </main>
    </div>
  )
}
