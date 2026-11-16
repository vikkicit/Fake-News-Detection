import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fake News Detection | AI-Powered News Intelligence",
  description: "Next-generation 3D AI Intelligence Environment for News Verification",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#020208] text-white selection:bg-cyan-500 selection:text-black">
        {children}
      </body>
    </html>
  );
}
