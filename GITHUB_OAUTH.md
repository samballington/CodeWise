# GitHub OAuth Setup Guide

This guide explains how to set up GitHub OAuth integration for CodeWise, allowing users to clone and work on their GitHub repositories.

## 1. Register GitHub OAuth App

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in the application details:
   - **Application name**: CodeWise (or your preferred name)
   - **Homepage URL**: `http://localhost:3000`
   - **Authorization callback URL**: `http://localhost:8000/oauth/callback`
   - **Application description**: AI development assistant for GitHub repositories

4. Click "Register application"
5. Copy the **Client ID** and **Client Secret**

## 2. Configure Environment Variables

Add the following to your `.env` file:

```env
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
```

## 3. OAuth Flow

### Frontend Integration
- Click "GitHub" button in the header
- OAuth popup opens for GitHub authorization
- User grants permissions (repo access)
- Callback redirects to frontend with session token
- Repository list loads automatically

### Backend Endpoints
- `GET /oauth/login` - Initiates OAuth flow
- `GET /oauth/callback` - Handles GitHub callback
- `GET /oauth/repos` - Lists user repositories
- `POST /oauth/clone` - Prepares repository for cloning
- `DELETE /oauth/logout` - Revokes session

## 4. Security Considerations

- GitHub tokens are stored server-side only
- CSRF protection via state parameter
- Session-based token management
- Automatic cleanup on logout
- Scoped permissions (repo access only)

## 5. Usage

1. Start CodeWise with OAuth configured
2. Click "GitHub" in the header
3. Authenticate with GitHub
4. Browse and select repositories
5. Click a repository to clone it
6. Work on the repository using CodeWise

## 6. Troubleshooting

**OAuth popup blocked**: Enable popups for localhost:3000
**Callback URL mismatch**: Ensure callback URL matches exactly
**Missing permissions**: Check OAuth app scopes include 'repo'
**Token errors**: Verify client ID and secret are correct

## 7. Production Deployment

For production deployment:
- Update callback URL to your domain
- Use HTTPS for all URLs
- Store tokens in encrypted database
- Implement token refresh logic
- Add rate limiting and monitoring 