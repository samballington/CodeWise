'use client'

import { useContextStore } from '@/lib/store'
import { useEffect, useState } from 'react'
import { Search, FileText, CheckCircle, AlertCircle } from 'lucide-react'

export const ContextPopup = () => {
  const { isGatheringContext, currentActivity } = useContextStore()
  const [showPopup, setShowPopup] = useState(false)
  const [fadeOut, setFadeOut] = useState(false)

  useEffect(() => {
    if (isGatheringContext && currentActivity) {
      setShowPopup(true)
      setFadeOut(false)
    } else if (!isGatheringContext && currentActivity?.type === 'complete') {
      // Show completion briefly then fade out
      setTimeout(() => {
        setFadeOut(true)
        setTimeout(() => setShowPopup(false), 500)
      }, 1500)
    }
  }, [isGatheringContext, currentActivity])

  if (!showPopup || !currentActivity) return null

  const getIcon = () => {
    switch (currentActivity.type) {
      case 'start':
        return <Search className="w-4 h-4 animate-pulse text-blue-500" />
      case 'search':
        return <FileText className="w-4 h-4 animate-pulse text-orange-500" />
      case 'complete':
        return currentActivity.chunksFound === 0 ? 
          <AlertCircle className="w-4 h-4 text-yellow-500" /> :
          <CheckCircle className="w-4 h-4 text-green-500" />
      default:
        return <Search className="w-4 h-4 text-gray-500" />
    }
  }

  const getBackgroundColor = () => {
    switch (currentActivity.type) {
      case 'start':
        return 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800'
      case 'search':
        return 'bg-orange-50 border-orange-200 dark:bg-orange-900/20 dark:border-orange-800'
      case 'complete':
        return currentActivity.chunksFound === 0 ?
          'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800' :
          'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800'
      default:
        return 'bg-gray-50 border-gray-200 dark:bg-gray-900/20 dark:border-gray-800'
    }
  }

  return (
    <div className={`fixed top-4 right-4 z-50 transition-all duration-500 ${
      fadeOut ? 'opacity-0 translate-y-2' : 'opacity-100 translate-y-0'
    }`}>
      <div className={`
        px-4 py-3 rounded-lg shadow-lg border backdrop-blur-sm
        ${getBackgroundColor()}
        max-w-sm
      `}>
        <div className="flex items-center space-x-3">
          {getIcon()}
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {currentActivity.message}
            </p>
            {currentActivity.query && (
              <p className="text-xs text-gray-600 dark:text-gray-400 truncate mt-1">
                Query: "{currentActivity.query}"
              </p>
            )}
            {currentActivity.type === 'complete' && currentActivity.sources && (
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Sources: {currentActivity.sources.length} areas analyzed
              </p>
            )}
          </div>
        </div>
        
        {/* Progress indicator for active searches */}
        {currentActivity.type === 'search' && (
          <div className="mt-2">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
              <div className="bg-orange-500 h-1 rounded-full animate-pulse" style={{ width: '60%' }}></div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 