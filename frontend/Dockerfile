# === STAGE 1: Build Stage ===
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install ALL dependencies (including dev dependencies)
RUN npm install

# Copy all source files
COPY . .

# Ensure public directory exists even if repo has none (so production copy succeeds)
RUN mkdir -p public

# Build the application
RUN npm run build

# === STAGE 2: Production Stage ===
FROM node:18-alpine as production

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Reuse modules from builder and prune dev deps to avoid lock mismatch
COPY --from=builder /app/node_modules ./node_modules
RUN npm prune --omit=dev && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Copy any other necessary runtime files
COPY next.config.js* ./

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 