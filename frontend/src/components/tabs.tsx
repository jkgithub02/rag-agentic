"use client";

interface TabsProps {
    active: string;
    items: Array<{ key: string; label: string }>;
    onChange: (key: string) => void;
}

export function Tabs({ active, items, onChange }: TabsProps) {
    return (
        <nav className="inline-flex rounded-2xl border border-[var(--line)] bg-white/70 p-1 shadow-sm backdrop-blur">
            {items.map((item) => {
                const isActive = active === item.key;
                return (
                    <button
                        key={item.key}
                        type="button"
                        className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${isActive
                                ? "bg-[var(--ink)] text-white"
                                : "text-[var(--ink)] hover:bg-[var(--paper)]"
                            }`}
                        onClick={() => onChange(item.key)}
                    >
                        {item.label}
                    </button>
                );
            })}
        </nav>
    );
}
