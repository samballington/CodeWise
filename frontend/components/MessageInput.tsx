'use client'

import React, { useState, KeyboardEvent, useRef, useEffect } from 'react'
import { Send, Square } from 'lucide-react'
import { useThemeStore } from '../lib/store'

interface MessageInputProps {
  onSendMessage: (message: string) => void
}

export default function MessageInput({ onSendMessage }: MessageInputProps) {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { isDarkMode } = useThemeStore()

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput('')
      setIsLoading(true)
      // Reset loading state after a delay
      setTimeout(() => setIsLoading(false), 1000)
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
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
    <div className="relative">
      <div className={`flex items-end gap-3 rounded-2xl p-2 focus-ring transition-all duration-200 border ${
        isDarkMode 
          ? 'bg-background-secondary border-border hover:border-primary/50' 
          : 'bg-gray-50 border-gray-200 hover:border-primary/50'
      }`}>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Message CodeWise..."
          className={`flex-1 bg-transparent resize-none border-none outline-none px-3 py-2 min-h-[44px] max-h-[200px] leading-6 transition-colors duration-200 ${
            isDarkMode 
              ? 'text-text-primary placeholder-text-secondary' 
              : 'text-gray-900 placeholder-gray-500'
          }`}
          disabled={isLoading}
          rows={1}
        />
        
        <button
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
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
      
      {/* Keyboard shortcut hint */}
      <div className={`mt-2 text-xs text-center opacity-60 transition-colors duration-200 ${
        isDarkMode ? 'text-text-secondary' : 'text-gray-500'
      }`}>
        Press Enter to send, Shift+Enter for new line
      </div>
    </div>
  )
} 