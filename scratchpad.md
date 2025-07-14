# CodeWise Development Scratchpad - Critical Fixes

## Current Critical Issues

### Issue 1: Code Viewer Too Narrow - Blue Empty Space
**Problem**: The Monaco Editor is taking only a small portion of the right side, leaving a large blue/dark empty area

**Root Cause Analysis**:
- CodeViewer component not filling available width
- Monaco Editor container not sized properly
- CSS layout issues in ProjectLayout or CodeViewer
- Grid/Flexbox not expanding to full width

**Solutions**:
1. **CodeViewer Sizing**: Ensure Monaco Editor takes full width/height of container
2. **Container CSS**: Fix parent container to use full available space
3. **Grid Layout**: Verify grid proportions are working correctly
4. **Monaco Config**: Check Monaco Editor sizing options

### Issue 2: GitHub Auth Disrupts Project View
**Problem**: Signing into GitHub closes/disrupts the Projects tab, forcing user back to Chat mode

**Root Cause**:
- Page navigation or state reset during OAuth flow
- Component unmounting during auth process
- State management not preserving project view mode

**Solutions**:
1. **Preserve View State**: Maintain project view during auth flow
2. **OAuth Handling**: Improve auth flow to not disrupt current view
3. **State Persistence**: Ensure view mode survives page refreshes/navigations

### Issue 3: Generic LLM System Prompt
**Problem**: Current system prompt is too generic, not code-focused, overly agreeable

**Requirements**:
- XML-based prompting structure
- Code-first expert persona
- Concise, helpful, not overly agreeable
- Technical depth and precision
- Direct, actionable responses

## Implementation Plan

### Phase 1: Fix Code Viewer Layout (HIGH PRIORITY)
1. **Investigate CodeViewer Component**:
   - Check Monaco Editor container sizing
   - Verify CSS classes and styling
   - Ensure proper width/height properties

2. **Fix Layout CSS**:
   - ProjectLayout grid proportions
   - CodeViewer container styling
   - Monaco Editor configuration

3. **Test Responsive Behavior**:
   - Ensure layout works at different screen sizes
   - Verify sidebar collapse/expand functionality

### Phase 2: Fix GitHub Auth Flow
1. **State Management**:
   - Preserve project view during auth
   - Prevent unnecessary component unmounting
   - Maintain current project selection

2. **OAuth Flow Improvement**:
   - Handle auth callback without disrupting UI
   - Keep user in same view mode after auth

### Phase 3: Improve LLM System Prompt
1. **XML Structure**: Use clear XML tags for different prompt sections
2. **Expert Persona**: Code-first, direct, technically precise
3. **Response Guidelines**: Concise, actionable, not overly agreeable

## Technical Details

### CodeViewer Layout Fix
```typescript
// Expected structure in CodeViewer.tsx
<div className="h-full w-full"> {/* Full container */}
  <MonacoEditor
    width="100%"
    height="100%"
    // Other props
  />
</div>
```

### CSS Investigation Areas
- `frontend/components/CodeViewer.tsx` - Monaco container
- `frontend/components/ProjectLayout.tsx` - Grid layout
- `frontend/app/globals.css` - Global styles
- Check for conflicting height/width constraints

### GitHub Auth State Preservation
```typescript
// In useProjectStore or main state management
- Preserve current view mode during auth
- Don't reset project selection on auth completion
- Handle auth callbacks gracefully
```

### New LLM System Prompt Structure
```xml
<role>
Expert software engineer and code architect
</role>

<expertise>
- Full-stack development
- System design and architecture  
- Code optimization and best practices
- Direct problem-solving approach
</expertise>

<response_guidelines>
- Be concise and actionable
- Provide specific code solutions
- Don't over-explain obvious concepts
- Challenge inefficient approaches
- Focus on practical implementation
</response_guidelines>

<communication_style>
- Direct and technical
- Solution-oriented
- Not overly agreeable or polite
- Assume user has technical competence
- Provide working code examples
</communication_style>

## Files to Update

### Layout/UI Fixes
- `frontend/components/CodeViewer.tsx` - Monaco sizing
- `frontend/components/ProjectLayout.tsx` - Grid layout
- `frontend/app/globals.css` - Style fixes if needed

### Auth Flow Fixes  
- `frontend/components/GitHubAuth.tsx` - Auth handling
- `frontend/lib/store.ts` - State preservation
- `frontend/app/page.tsx` - View mode persistence

### LLM System Prompt
- `backend/agent.py` - Update system prompt with XML structure
- Focus on code-first expert persona

## Testing Plan
1. **Layout Testing**: Verify code viewer fills right side completely
2. **Auth Testing**: Sign in/out while in project view, ensure no disruption
3. **LLM Testing**: Test new prompt with code-related queries
4. **Responsive Testing**: Verify layouts work at different screen sizes

## Expected Outcomes
- Code viewer uses full available width (no blue empty space)
- GitHub auth doesn't disrupt project view
- LLM responses are more direct, code-focused, and expert-level
- Better overall user experience for code browsing and AI interaction 