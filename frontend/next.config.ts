import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // Enables minimal self-contained Docker image
};

export default nextConfig;
