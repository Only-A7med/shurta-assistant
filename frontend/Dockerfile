# The web interface: built with Node, served as static files by nginx, which
# also proxies /api to the API container so the browser sees one origin.
FROM node:24-alpine AS build

WORKDIR /app

# Dependencies first, so a change to the source does not reinstall them.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM nginx:1.29-alpine AS runtime

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
