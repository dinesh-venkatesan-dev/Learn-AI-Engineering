// Hash routes: #/ · #/t/<track> · #/t/<track>/<module> · …/lab/<lab id> · …/path[/<step>]
import { useEffect, useState } from "react";

export type Route =
  | { page: "home" }
  | { page: "track"; track: string }
  | { page: "module"; track: string; module: string }
  | { page: "lab"; track: string; module: string; lab: string }
  | { page: "path"; track: string; module: string; step: string | null };

export function parse(hash: string): Route {
  const [kind, track, module, sub, id] = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (kind !== "t" || !track) return { page: "home" };
  if (!module) return { page: "track", track };
  if (sub === "lab" && id) return { page: "lab", track, module, lab: id };
  if (sub === "path") return { page: "path", track, module, step: id ?? null };
  return { page: "module", track, module };
}

export const href = {
  home: () => "#/",
  track: (track: string) => `#/t/${track}`,
  module: (track: string, module: string) => `#/t/${track}/${module}`,
  lab: (track: string, module: string, lab: string) => `#/t/${track}/${module}/lab/${lab}`,
  path: (track: string, module: string, step?: string) => `#/t/${track}/${module}/path${step ? `/${step}` : ""}`,
};

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(location.hash));
  useEffect(() => {
    const onChange = () => { setRoute(parse(location.hash)); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
