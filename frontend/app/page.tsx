'use client'

import React, { useEffect, useState } from 'react'
import ChatInterface from '../components/ChatInterface'
import ChatOverlay from '../components/ChatOverlay'
import { useWebSocket } from '../hooks/useWebSocket'
import { useChatStore, useThemeStore } from '../lib/store'
import GitHubAuth from '../components/GitHubAuth'
import { ProjectLayout } from '../components/ProjectLayout'
import APIProviderToggle from '../components/APIProviderToggle'

export default function Home() {
  const { connected, sendMessage } = useWebSocket()
  const { messages } = useChatStore()
  const { isDarkMode, toggleTheme } = useThemeStore()
  const [showGitHub, setShowGitHub] = useState(false)
  const [showProjects, setShowProjects] = useState(false)
  const [githubSession, setGithubSession] = useState<string | null>(null)
  const [showChatOverlay, setShowChatOverlay] = useState(false)

  // Apply theme to document
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  return (
    <main className={`flex min-h-screen flex-col transition-colors duration-300 ${
      isDarkMode ? 'bg-background-primary' : 'bg-gray-50'
    }`}>
      <header className={`border-b shadow-lg transition-colors duration-300 ${
        isDarkMode 
          ? 'bg-background-secondary border-border' 
          : 'bg-white border-gray-200'
      }`}>
        <div className="container mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">CW</span>
            </div>
            <h1 className={`text-xl font-semibold transition-colors duration-300 ${
              isDarkMode ? 'text-text-primary' : 'text-gray-900'
            }`}>CodeWise</h1>
          </div>
          <div className="flex items-center gap-4">
            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-all duration-200 hover:scale-105 ${
                isDarkMode 
                  ? 'bg-slate-700 hover:bg-slate-600 text-yellow-400' 
                  : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
              }`}
              title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {isDarkMode ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              )}
            </button>

            {/* API Provider Toggle */}
            <APIProviderToggle />

            {/* GitHub toggle */}
            <button
              onClick={() => setShowGitHub(!showGitHub)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                showGitHub
                  ? 'bg-primary text-white'
                  : isDarkMode
                  ? 'bg-slate-700 text-text-primary hover:bg-slate-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              GitHub
            </button>
            
            {/* Chat toggle */}
            <button
              onClick={() => setShowChatOverlay(!showChatOverlay)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                showChatOverlay
                  ? 'bg-primary text-white'
                  : isDarkMode
                  ? 'bg-slate-700 text-text-primary hover:bg-slate-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Chat
            </button>

            {/* Projects toggle */}
            <button
              onClick={() => setShowProjects(!showProjects)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                showProjects
                  ? 'bg-primary text-white'
                  : isDarkMode
                  ? 'bg-slate-700 text-text-primary hover:bg-slate-600'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Projects
            </button>
            
            <div className="flex items-center gap-2">
              <div
                className={`status-icon ${
                  connected ? 'status-connected' : 'status-disconnected'
                }`}
              />
              <span
                className={`text-sm transition-colors duration-300 ${
                  isDarkMode ? 'text-text-secondary' : 'text-gray-600'
                }`}
              >
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>
      
      <div className="flex-1 flex min-h-0">
        {showProjects ? (
          <div className="flex-1 flex min-h-0">
            <ProjectLayout />
            {showGitHub && (
              <div className="w-96 border-l border-gray-200 bg-white p-4 overflow-y-auto">
                <GitHubAuth onAuthSuccess={(session) => {
                  setGithubSession(session)
                  setShowGitHub(false) // Close GitHub panel after successful auth
                }} />
              </div>
            )}
          </div>
        ) : showGitHub ? (
          <div className="flex-1 p-6">
            <GitHubAuth onAuthSuccess={(session) => setGithubSession(session)} />
          </div>
        ) : (
          <ChatInterface />
        )}
      </div>
      {/* Chat overlay */}
      {showChatOverlay && <ChatOverlay onClose={() => setShowChatOverlay(false)} />}
    </main>
  )
}