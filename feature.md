# File Tree & Code Viewer Feature

## Overview
This feature adds a comprehensive project file explorer and code viewer to CodeWise, allowing users to browse cloned repositories with an aesthetic file tree interface and view code files with syntax highlighting.

## Key Features

### 1. Project Management
- **Project Listing**: Displays all cloned repositories in the workspace
- **Project Selection**: Easy switching between different projects
- **Project Metadata**: Shows last modified date and total size
- **Auto-refresh**: Automatically loads projects on component mount

### 2. File Tree Navigation
- **Hierarchical View**: Nested folder structure with expand/collapse
- **File Type Icons**: Visual indicators for different file types and languages
- **Smart Sorting**: Directories first, then files, alphabetically sorted
- **Path Breadcrumbs**: Clear indication of current project context

### 3. Code Viewer
- **Monaco Editor**: Professional code editor with syntax highlighting
- **Multi-tab Interface**: Open multiple files simultaneously with tabs
- **Language Detection**: Automatic language detection based on file extension
- **Binary File Handling**: Graceful handling of non-text files
- **Read-only Mode**: Safe viewing without accidental modifications

### 4. UI/UX Enhancements
- **Collapsible Sidebar**: Toggle file tree visibility
- **Responsive Design**: Works on different screen sizes
- **Loading States**: Smooth loading indicators
- **Error Handling**: User-friendly error messages
- **Hover Effects**: Interactive feedback on all clickable elements

## Technical Implementation

### Backend API Endpoints

#### GET /projects
- Lists all projects (directories) in the workspace
- Returns project metadata (name, path, modified date, size)
- Sorted by last modified date (newest first)

#### GET /projects/{project_name}/tree
- Returns hierarchical file tree structure for a project
- Includes file metadata (size, type, modified date)
- Filters out hidden files (starting with '.')
- Recursive directory traversal with proper error handling

#### GET /projects/{project_name}/file
- Serves individual file content with metadata
- Handles both text and binary files
- Includes MIME type detection
- Security validation to prevent directory traversal

### Frontend Components

#### ProjectSelector
- Displays available projects in card format
- Shows project metadata (size, last modified)
- Handles project switching with file tree loading
- Loading and error states

#### FileTree
- Recursive tree component with expand/collapse
- File type icons using React Icons
- Click handlers for file opening
- Proper keyboard navigation support

#### CodeViewer
- Monaco Editor integration with language detection
- Tab-based interface for multiple files
- Binary file detection and handling
- Syntax highlighting for 20+ languages

#### ProjectLayout
- Main layout component combining all parts
- Collapsible sidebar functionality
- Responsive design with proper spacing
- Header with toggle controls

### State Management
- **Zustand Store**: Centralized state for projects, files, and UI
- **API Integration**: Async actions for data fetching
- **Error Handling**: Proper error state management
- **Loading States**: Loading indicators throughout the app

## File Structure

### Backend
```
backend/
├── routers/
│   ├── __init__.py
│   └── projects.py          # Project management endpoints
└── main.py                  # Updated with new router
```

### Frontend
```
frontend/
├── types/
│   └── project.ts           # TypeScript interfaces
├── lib/
│   └── projectStore.ts      # Zustand store
├── components/
│   ├── FileIcon.tsx         # File type icons
│   ├── ProjectSelector.tsx  # Project selection
│   ├── FileTree.tsx         # File tree navigation
│   ├── CodeViewer.tsx       # Code viewing with Monaco
│   └── ProjectLayout.tsx    # Main layout
└── app/
    └── page.tsx             # Updated main page
```

## Dependencies Added

### Frontend
- `@monaco-editor/react`: Professional code editor
- `react-icons`: Comprehensive icon library

### Backend
- Enhanced `pathlib` usage for file operations
- `mimetypes` for file type detection
- `datetime` for timestamp formatting

## Security Considerations

1. **Path Validation**: All file paths validated to prevent directory traversal
2. **Workspace Isolation**: Operations restricted to workspace directory
3. **Binary File Detection**: Safe handling of non-text files
4. **Error Boundaries**: Graceful error handling without exposing internals

## Performance Optimizations

1. **Lazy Loading**: File tree branches loaded on demand
2. **Caching**: File contents cached in store to avoid re-fetching
3. **Virtualization**: Ready for large file lists (can be added later)
4. **Debouncing**: Search operations debounced (future enhancement)

## Recent Enhancements (Latest Update)

### Workspace Root File Support
- **Special "workspace" Project**: Files created directly in workspace root now appear as a special "workspace" project
- **Home Icon**: Workspace project distinguished with a Home icon (orange) vs regular Folder icon (blue)
- **AI Integration**: When AI creates files like `index.html`, `style.css`, they're immediately visible in the project explorer
- **Backend Handling**: Modified ProjectService to detect and serve workspace root files

### Improved Layout Proportions  
- **Narrower Sidebar**: Reduced sidebar width from 320px to 256px (20% space savings)
- **Full-Width Code Viewer**: Monaco Editor now properly fills all available horizontal space
- **Layout Fixes**: Added explicit width/height constraints and min-h-0 for proper flex behavior
- **Container Optimization**: Code viewer container uses full parent dimensions
- **Responsive Design**: Maintains collapsible functionality with improved proportions

### Enhanced GitHub Integration
- **Non-Disruptive Auth**: GitHub authentication now appears as sidebar panel in Project view
- **State Preservation**: Project view remains active during authentication flow
- **Seamless Workflow**: Auto-closes GitHub panel after successful authentication
- **Better UX**: No more view mode switching when accessing GitHub features

### Improved AI Agent System Prompt
- **XML-Structured Prompting**: Clear role definition, expertise areas, and response guidelines
- **Code-First Expert Persona**: Direct, technically precise, solution-oriented responses
- **Professional Communication**: Assumes technical competence, challenges inefficient approaches
- **Focused Capabilities**: Explicit tool descriptions and execution methodology
- **Production-Ready Code**: Emphasis on working, tested solutions with proper error handling

## User Experience

### Navigation Flow
1. User clicks "Projects" button in header
2. ProjectSelector shows available cloned repositories AND workspace files
3. **Workspace Files**: If AI creates files in root, they appear under "Workspace Files" project
4. Clicking a project loads its file tree
5. Clicking files opens them in the code viewer
6. Multiple files can be open in tabs
7. Sidebar can be collapsed for more viewing space

### Visual Design
- **Clean Interface**: Minimal, professional appearance
- **Consistent Icons**: File type recognition at a glance
- **Smooth Animations**: Transitions for expand/collapse and hover states
- **Proper Spacing**: Adequate padding and margins for readability
- **Color Coding**: File types distinguished by icon colors

## Future Enhancements

1. **Search Functionality**: File and content search within projects
2. **File Operations**: Create, rename, delete files
3. **Git Integration**: Show git status in file tree
4. **Diff Viewer**: Compare file versions
5. **Minimap**: Code overview for large files
6. **Themes**: Dark/light mode for code editor
7. **Bookmarks**: Save frequently accessed files

## Testing Strategy

1. **Backend Testing**: API endpoint validation
2. **Frontend Testing**: Component unit tests
3. **Integration Testing**: Full workflow testing
4. **Error Scenarios**: Network failures, invalid files
5. **Performance Testing**: Large repository handling

## Deployment Notes

- Feature is behind a toggle button in the main interface
- Backward compatible with existing chat functionality
- No database changes required
- Uses existing workspace directory structure

This feature significantly enhances CodeWise's usability by providing a professional code browsing experience similar to modern IDEs while maintaining the simplicity and focus of the original chat interface. 