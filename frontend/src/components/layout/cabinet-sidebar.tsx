"use client"

import React from 'react'
import { Cpu, BookOpen, Terminal, Archive, Clock, Calendar } from 'lucide-react'
import { cn } from "@/lib/utils"

export type CabinetId = 'today' | 'yesterday' | 'week' | 'older' | 'saved'

export interface Cabinet {
  id: CabinetId
  name: string
  icon: React.ReactNode
  description: string
  color: string
}

export const CABINETS: Cabinet[] = [
  { 
    id: 'today', 
    name: 'Current Discussions', 
    icon: <Cpu size={18} />, 
    description: 'Active Context',
    color: 'bg-stone-800'
  },
  { 
    id: 'yesterday', 
    name: 'Recent History', 
    icon: <Clock size={18} />, 
    description: 'Last 24 Hours',
    color: 'bg-stone-700'
  },
  { 
    id: 'week', 
    name: 'Weekly Overview', 
    icon: <Calendar size={18} />, 
    description: 'Last 7 Days',
    color: 'bg-stone-600'
  },
  { 
    id: 'older', 
    name: 'Archived', 
    icon: <Archive size={18} />, 
    description: 'Older Discussions',
    color: 'bg-stone-500'
  },
]

interface CabinetSidebarProps {
  activeCabinet: CabinetId
  setActiveCabinet: (id: CabinetId) => void
  counts: Record<CabinetId, number>
}

export function CabinetSidebar({ activeCabinet, setActiveCabinet, counts }: CabinetSidebarProps) {
  return (
    <div className="w-64 h-full bg-[#0f291e] border-r border-[#1a4533] flex flex-col relative z-20 shadow-2xl shrink-0">
      <div className="p-6 border-b border-[#1a4533]/50">
        <h1 className="text-xl font-serif tracking-wider text-emerald-100/80">Personal Archive</h1>
        <p className="text-[10px] text-emerald-600 mt-1 uppercase tracking-[0.2em]">Knowledge Base</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-2">
        {CABINETS.map((cabinet) => (
          <button
            key={cabinet.id}
            onClick={() => setActiveCabinet(cabinet.id)}
            className={cn(
              "w-full flex items-center gap-3 p-3 rounded-lg transition-all duration-300 border group text-left",
              activeCabinet === cabinet.id 
                ? "bg-[#1a4533] text-white border-emerald-800 shadow-lg shadow-emerald-900/20" 
                : "hover:bg-[#143628] text-emerald-200/70 border-transparent hover:border-[#1a4533]"
            )}
          >
            <div className={cn(
              "p-2 rounded-md transition-colors",
              activeCabinet === cabinet.id ? "bg-emerald-900/50 text-emerald-100" : "bg-[#0a1f16] text-emerald-700 group-hover:text-emerald-400"
            )}>
              {cabinet.icon}
            </div>
            <div className="flex-1 min-w-0">
              <span className="text-sm font-medium block truncate">{cabinet.name}</span>
              <span className="text-[10px] opacity-50 uppercase tracking-wider">{cabinet.description}</span>
            </div>
            <span className={cn(
              "text-xs font-mono transition-opacity",
              activeCabinet === cabinet.id ? "opacity-100 text-emerald-400" : "opacity-30"
            )}>
              {counts[cabinet.id] || 0}
            </span>
          </button>
        ))}
      </div>

      <div className="p-4 border-t border-[#1a4533]/50 bg-[#0a1f16]/30">
        <div className="flex items-center gap-2 text-emerald-800/50 text-[10px] font-mono uppercase">
          <div className="w-2 h-2 rounded-full bg-emerald-900 animate-pulse" />
          System Online
        </div>
      </div>
    </div>
  )
}
