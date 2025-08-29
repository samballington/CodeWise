'use client'

import React, { useState, KeyboardEvent, useRef, useEffect, useCallback } from 'react'
import { Send, Square, ChevronDown } from 'lucide-react'
import { useThemeStore, useModelStore } from '../lib/store'
import { useProjectStore } from '../lib/projectStore'
import ModelSelector from './ModelSelector'

interface MessageInputProps {
  onSendMessage: (message: string, mentionedProjects?: string[], selectedModel?: string) => void
  disabled?: boolean
}

interface MentionDropdownProps {
  projects: Array<{name: string, is_workspace_root?: boolean}>
  position: { top: number, left: number }
  onSelect: (project: string) => void
  onClose: () => void
  filter: string
}

const MentionDropdown: React.FC<MentionDropdownProps> = ({ 
  projects, 
  position, 
  onSelect, 
  onClose, 
  filter 
}) => {
  const { isDarkMode } = useThemeStore()
  const dropdownRef = useRef<HTMLDivElement>(null)
  
  // Filter projects based on current input
  const filteredProjects = projects.filter(project => 
    project.name.toLowerCase().includes(filter.toLowerCase())
  ).slice(0, 5) // Limit to 5 results
  
  // Debug: Log filtering
  console.log('🔍 MentionDropdown: All projects:', projects.map(p => p.name))
  console.log('🔍 MentionDropdown: Filter:', filter)
  console.log('🔍 MentionDropdown: Filtered projects:', filteredProjects.map(p => p.name))
  
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        onClose()
      }
    }
    
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [onClose])
  
  if (filteredProjects.length === 0) return null
  
  return (
    <div
      ref={dropdownRef}
      className={`absolute z-50 min-w-48 max-w-64 rounded-lg shadow-lg border ${
        isDarkMode 
          ? 'bg-background-secondary border-border' 
          : 'bg-white border-gray-200'
      }`}
      style={{ top: position.top, left: position.left }}
    >
      <div className="py-1">
        {filteredProjects.map((project, index) => (
          <button
            key={project.name}
            className={`w-full px-3 py-2 text-left text-sm hover:bg-opacity-80 transition-colors ${
              isDarkMode 
                ? 'hover:bg-slate-600 text-text-primary' 
                : 'hover:bg-gray-100 text-gray-900'
            }`}
            onClick={() => onSelect(project.name)}
          >
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                project.is_workspace_root ? 'bg-orange-500' : 'bg-blue-500'
              }`} />
              <span className="font-medium">
                {project.is_workspace_root ? 'workspace' : project.name}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

export default function MessageInput({ onSendMessage, disabled = false }: MessageInputProps) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showMentionDropdown, setShowMentionDropdown] = useState(false)
  const [mentionPosition, setMentionPosition] = useState({ top: 0, left: 0 })
  const [currentMentionFilter, setCurrentMentionFilter] = useState('')
  const [mentionStartIndex, setMentionStartIndex] = useState(-1)
  
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isDarkMode } = useThemeStore()
  const { selectedModel, setSelectedModel } = useModelStore()
  const { projects, fetchProjects } = useProjectStore()

  // Fetch projects on component mount
  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  // Parse mentions from input text
  const parseMentions = useCallback((text: string): string[] => {
    const mentionRegex = /@(\w+)/g
    const mentions: string[] = []
    let match
    
    while ((match = mentionRegex.exec(text)) !== null) {
      const mentionedProject = match[1]
      // Check if mentioned project exists
      const projectExists = projects.some(p => 
        (p.is_workspace_root && mentionedProject === 'workspace') || 
        p.name === mentionedProject
      )
      if (projectExists) {
        mentions.push(mentionedProject)
      }
    }
    
    return mentions
  }, [projects])

  // Handle input changes and detect @ mentions
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value
    const cursorPosition = e.target.selectionStart
    
    setInput(newValue)
    
    // Check for @ mention trigger - be more aggressive in detection
    const textBeforeCursor = newValue.substring(0, cursorPosition)
    const mentionMatch = textBeforeCursor.match(/@(\w*)$/)
    
    if (mentionMatch) {
      const filter = mentionMatch[1]
      const mentionStart = textBeforeCursor.lastIndexOf('@')
      
      setCurrentMentionFilter(filter)
      setMentionStartIndex(mentionStart)
      setShowMentionDropdown(true)
      
      // Calculate dropdown position more accurately
      if (textareaRef.current) {
        const textarea = textareaRef.current
        const textareaRect = textarea.getBoundingClientRect()
        
        // Position dropdown below the textarea with some offset
        setMentionPosition({
          top: textareaRect.bottom + 8,
          left: textareaRect.left
        })
      }
    } else {
      setShowMentionDropdown(false)
    }
  }, [])

  // Handle mention selection
  const handleMentionSelect = useCallback((projectName: string) => {
    
    if (textareaRef.current && mentionStartIndex >= 0) {
      const beforeMention = input.substring(0, mentionStartIndex)
      const afterMention = input.substring(textareaRef.current.selectionStart)
      const newInput = `${beforeMention}@${projectName} ${afterMention}`
      
      setInput(newInput)
      setShowMentionDropdown(false)
      
      // Focus back on textarea and position cursor
      setTimeout(() => {
        if (textareaRef.current) {
          const newCursorPosition = beforeMention.length + projectName.length + 2
          textareaRef.current.focus()
          textareaRef.current.setSelectionRange(newCursorPosition, newCursorPosition)
        }
      }, 0)
    }
  }, [input, mentionStartIndex])

  // Render text with highlighted mentions - simplified approach
  const renderHighlightedText = useCallback((text: string) => {
    const mentions = parseMentions(text)
    if (mentions.length === 0) return text

    // Create a simple highlighted version
    let highlightedText = text
    mentions.forEach(mention => {
      const pattern = new RegExp(`@${mention}(?=\\s|$)`, 'g')
      highlightedText = highlightedText.replace(
        pattern, 
        `<mark class="mention-highlight">@${mention}</mark>`
      )
    })
    
    return highlightedText
  }, [parseMentions])

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      const mentions = parseMentions(input.trim())
      onSendMessage(input.trim(), mentions.length > 0 ? mentions : undefined, selectedModel)
      setInput('')
      setIsLoading(true)
      setShowMentionDropdown(false)
      // Reset loading state after a delay
      setTimeout(() => setIsLoading(false), 1000)
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (showMentionDropdown) {
        setShowMentionDropdown(false)
      } else {
      handleSend()
      }
    } else if (e.key === 'Escape' && showMentionDropdown) {
      setShowMentionDropdown(false)
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  return (
    <>
    <div className="relative">
      {/* Model Selector */}
      <div className="mb-3">
        <ModelSelector
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          disabled={isLoading || disabled}
        />
      </div>
      
      <div className={`flex items-end gap-3 rounded-2xl p-2 focus-ring transition-all duration-200 border ${
        isDarkMode 
          ? 'bg-background-secondary border-border hover:border-primary/50' 
          : 'bg-gray-50 border-gray-200 hover:border-primary/50'
      }`}>
          <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={input}
              onChange={handleInputChange}
          onKeyPress={handleKeyPress}
              placeholder="Message CodeWise... (use @project to specify repository)"
              className={`w-full bg-transparent resize-none border-none outline-none px-3 py-2 min-h-[44px] max-h-[200px] leading-6 transition-colors duration-200 ${
            isDarkMode 
              ? 'text-text-primary placeholder-text-secondary' 
              : 'text-gray-900 placeholder-gray-500'
          }`}
              disabled={isLoading || disabled}
          rows={1}
              style={{ position: 'relative', zIndex: 2 }}
            />
            {/* Highlighted overlay for mentions */}
            <div 
              className={`absolute inset-0 px-3 py-2 pointer-events-none overflow-hidden whitespace-pre-wrap break-words leading-6 ${
                isDarkMode ? 'text-transparent' : 'text-transparent'
              }`}
              style={{ 
                zIndex: 1,
                fontSize: 'inherit',
                fontFamily: 'inherit',
                minHeight: '44px',
                maxHeight: '200px'
              }}
              dangerouslySetInnerHTML={{ 
                __html: renderHighlightedText(input).replace(
                  /mention-highlight/g, 
                  `mention-highlight inline px-1 rounded text-xs font-medium ${
                    isDarkMode ? 'bg-blue-600 text-blue-100' : 'bg-blue-100 text-blue-800'
                  }`
                ) 
              }}
            />
          </div>
        
        <button
          onClick={handleSend}
            disabled={!input.trim() || isLoading || disabled}
          className={`flex items-center justify-center w-10 h-10 rounded-xl transition-all duration-200 ${
            input.trim() && !isLoading
              ? 'bg-primary text-white hover:bg-primary/90 hover:scale-105 shadow-lg'
              : isDarkMode
              ? 'bg-border text-text-secondary cursor-not-allowed opacity-50'
              : 'bg-gray-300 text-gray-500 cursor-not-allowed opacity-50'
          }`}
        >
          {isLoading ? (
            <Square className="w-4 h-4" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>
      
      {/* Character count or status */}
      {input.length > 1000 && (
        <div className={`absolute -top-6 right-0 text-xs transition-colors duration-200 ${
          isDarkMode ? 'text-text-secondary' : 'text-gray-500'
        }`}>
          {input.length}/2000
        </div>
      )}
      
      <div className={`mt-2 text-xs text-center opacity-60 transition-colors duration-200 ${
        isDarkMode ? 'text-text-secondary' : 'text-gray-500'
      }`}>
          Press Enter to send, Shift+Enter for new line. Use @project to specify repository.
        </div>

        {/* Mention Dropdown */}
        {showMentionDropdown && (
          <MentionDropdown
            projects={projects}
            position={mentionPosition}
            onSelect={handleMentionSelect}
            onClose={() => setShowMentionDropdown(false)}
            filter={currentMentionFilter}
          />
        )}
      </div>
      
      {/* Global CSS for mention highlighting */}
      <style jsx global>{`
        .mention-highlight {
          background-color: ${isDarkMode ? '#1d4ed8' : '#dbeafe'} !important;
          color: ${isDarkMode ? '#dbeafe' : '#1d4ed8'} !important;
          padding: 2px 4px !important;
          border-radius: 4px !important;
          font-weight: 500 !important;
        }
      `}</style>
    </>
  )
} 