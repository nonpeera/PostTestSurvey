# ใช้ Node version ที่เสถียรกว่า
FROM node:20-alpine

# ติดตั้ง dependencies ที่จำเป็น
RUN apk add --no-cache \
    curl \
    wget \
    git \
    python3 \
    make \
    g++ \
    && rm -rf /var/cache/apk/*

# ตั้ง working directory
WORKDIR /app

# Copy package files ก่อน
COPY package*.json ./

# ล้าง npm cache และ install dependencies
RUN npm cache clean --force
RUN npm install --verbose

# Copy source code
COPY . .

# สร้าง non-root user เพื่อความปลอดภัย
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# เปลี่ยน ownership ของไฟล์
RUN chown -R nextjs:nodejs /app
USER nextjs

# เปิด port 3000
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# รันแอป
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]