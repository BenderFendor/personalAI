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

  return (
    <Accordion type="single" collapsible className="w-full mb-2">
      <AccordionItem value="thinking" className="border rounded-md px-2">
        <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
          <div className="flex items-center gap-2">
            <BrainCircuit className="h-4 w-4" />
            <span>Thinking Process</span>
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
