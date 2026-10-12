// Light/dark theme: follows the system until the user picks one; the choice is remembered in this browser.
import { useSyncExternalStore } from "react";

const KEY = "rag-course:theme";
const listeners = new Set<() => void>();
let chosen: "light" | "dark" | null = null;   // this visit's choice, even if storage is blocked

function stored(): "light" | "dark" | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function current(): boolean {
  const choice = chosen ?? stored();
  return choice ? choice === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function apply() {
  document.documentElement.classList.toggle("dark", current());
}
apply();

export function useTheme() {
  const dark = useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, current);
  const toggle = () => {
    chosen = dark ? "light" : "dark";
    try {
      localStorage.setItem(KEY, chosen);
    } catch {
      /* storage blocked: the theme still flips for this visit */
    }
    apply();
    listeners.forEach((l) => l());
  };
  return { dark, toggle };
}
