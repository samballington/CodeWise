'use client'

import React, { useState, useRef, useEffect } from 'react'
import { useWebSocket } from '../hooks/useWebSocket'
import { useChatStore, useThemeStore } from '../lib/store'
import { useProjectStore } from '../lib/projectStore'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import { ContextPopup } from './ContextPopup'
import { ChevronDown, Folder, FolderOpen, Trash2 } from 'lucide-react'

export default function ChatInterface() {
  const { sendMessage } = useWebSocket()
  const { messages } = useChatStore()
  const { isDarkMode } = useThemeStore()
  const { projects, fetchProjects, deleteProject } = useProjectStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showProjects, setShowProjects] = useState(false)
  const projectsDropdownRef = useRef<HTMLDivElement>(null)

  // Indexer ready state
  const [indexReady, setIndexReady] = useState<boolean>(true)

  // Poll indexer status on mount and fetch projects
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/indexer/status')
        const json = await res.json()
        setIndexReady(json.ready)
      } catch {
        setIndexReady(true) // fail open
      }
    }
    fetchStatus()
    fetchProjects() // Fetch projects when component mounts
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [fetchProjects])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Close projects dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (projectsDropdownRef.current && !projectsDropdownRef.current.contains(event.target as Node)) {
        setShowProjects(false)
      }
    }
    
    if (showProjects) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showProjects])

  // Enhanced sendMessage handler that supports project mentions
  const handleSendMessage = (message: string, mentionedProjects?: string[]) => {
    // If mentionedProjects are provided, include them in the message context
    const messageWithContext = {
      content: message,
      mentionedProjects: mentionedProjects || []
    }
    
    console.log('Sending message with context:', messageWithContext)
    sendMessage(messageWithContext)
  }

  // Handle project deletion
  const handleDeleteProject = async (projectName: string, event: React.MouseEvent) => {
    event.stopPropagation() // Prevent dropdown from closing
    if (confirm(`Are you sure you want to delete the project "${projectName}"? This will remove all indexed data for this project.`)) {
      await deleteProject(projectName)
    }
  }

  return (
    <div className="flex flex-col h-full w-full max-w-none mx-0 relative chat-page-padding">
      {/* Context Popup - positioned absolutely for overlay effect */}
      <ContextPopup />
      
      {/* Projects Indicator */}
      {projects.length > 0 && (
        <div className="px-4 pt-4 pb-2">
          <div className="relative" ref={projectsDropdownRef}>
            <button
              onClick={() => setShowProjects(!showProjects)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                isDarkMode 
                  ? 'bg-background-secondary hover:bg-slate-600 text-text-primary border border-border' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700 border border-gray-200'
              }`}
            >
              <Folder className="w-4 h-4" />
              <span>{projects.length} indexed project{projects.length !== 1 ? 's' : ''}</span>
              <ChevronDown className={`w-4 h-4 transition-transform ${showProjects ? 'rotate-180' : ''}`} />
            </button>
            
            {showProjects && (
              <div className={`absolute top-full left-0 mt-2 w-64 rounded-lg shadow-lg border z-50 ${
                isDarkMode 
                  ? 'bg-background-secondary border-border' 
                  : 'bg-white border-gray-200'
              }`}>
                <div className="p-2 max-h-48 overflow-y-auto">
                  {projects.map((project, index) => (
                    <div
                      key={project.name}
                      className={`flex items-center gap-2 px-3 py-2 rounded text-sm group ${
                        isDarkMode 
                          ? 'text-text-primary hover:bg-slate-600' 
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <div className={`w-2 h-2 rounded-full ${
                        project.is_workspace_root ? 'bg-orange-500' : 'bg-blue-500'
                      }`} />
                      <span className="font-medium truncate flex-1">
                        {project.is_workspace_root ? 'workspace' : project.name}
                      </span>
                      <button
                        onClick={(e) => handleDeleteProject(project.name, e)}
                        className={`opacity-0 group-hover:opacity-100 p-1 rounded transition-all hover:bg-red-500 hover:text-white ${
                          isDarkMode ? 'text-gray-400 hover:bg-red-500' : 'text-gray-500 hover:bg-red-500'
                        }`}
                        title="Delete project index"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <div className="w-16 h-16 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center mx-auto mb-6">
                <span className="text-white font-bold text-xl">CW</span>
              </div>
              <h2 className={`text-2xl font-semibold mb-3 transition-colors duration-200 ${
                isDarkMode ? 'text-text-primary' : 'text-gray-900'
              }`}>
                Welcome to CodeWise
              </h2>
              <p className={`text-lg leading-relaxed transition-colors duration-200 ${
                isDarkMode ? 'text-text-secondary' : 'text-gray-600'
              }`}>
                Your AI development assistant is ready to help. Ask me to read files, write code, run tests, or help with any development task.
              </p>
              <div className={`mt-6 text-sm transition-colors duration-200 ${
                isDarkMode ? 'text-text-secondary' : 'text-gray-600'
              }`}>
                <p className="mb-2">Try asking:</p>
                <div className="space-y-1 text-accent">
                  <p>"@SWE_Project Explain the main functionality"</p>
                  <p>"@workspace How does authentication work?"</p>
                  <p>"Show me the main components in this codebase"</p>
                </div>
                <div className={`mt-4 text-xs px-3 py-2 rounded-lg ${
                  isDarkMode ? 'bg-slate-800 text-slate-300' : 'bg-blue-50 text-blue-700'
                }`}>
                  💡 <strong>Tip:</strong> Use <code className="font-mono">@project</code> mentions to specify which repository you want to analyze
                </div>
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input Area */}
      <div className="border-t border-border/30">
        {!indexReady && (
          <div className="p-2 text-center text-sm bg-yellow-100 text-yellow-800">
            🔄 Indexer is still building the code index. Answers may be incomplete for a minute…
          </div>
        )}
        <div className="w-full px-4 py-4">
          <MessageInput onSendMessage={handleSendMessage} disabled={!indexReady} />
        </div>
      </div>
    </div>
  )
} 