"use client"

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { BrainCircuit } from "lucide-react"

interface ThinkingAccordionProps {
  thinking: string
}

export function ThinkingAccordion({ thinking }: ThinkingAccordionProps) {
  if (!thinking) return null

  // Show the last part of the thinking process to simulate streaming
  const preview = thinking.length > 180 ? "..." + thinking.slice(-180) : thinking

  return (
    <Accordion type="single" collapsible className="w-full mb-2">
      <AccordionItem value="thinking" className="border rounded-md px-2">
        <AccordionTrigger className="group py-2 text-sm text-muted-foreground hover:no-underline">
          <div className="flex flex-col items-start gap-1 w-full text-left">
            <div className="flex items-center gap-2">
              <BrainCircuit className="h-4 w-4" />
              <span>Thinking Process</span>
            </div>
            <span className="text-xs font-mono opacity-75 w-full block group-data-[state=open]:hidden line-clamp-2 whitespace-pre-wrap">
              {preview}
            </span>
          </div>
        </AccordionTrigger>
        <AccordionContent>
          <div className="text-sm text-muted-foreground whitespace-pre-wrap p-2 bg-muted/50 rounded-md font-mono">
            {thinking}
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
