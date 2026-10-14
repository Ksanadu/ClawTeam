"use client";

import { ImagePlus, SendHorizonal, X } from "lucide-react";
import { useRef, useState } from "react";

import type { ChatAttachment } from "@/lib/api";

const MAX_IMAGES = 4;
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export function ChatInput({
  disabled,
  onSend
}: {
  disabled: boolean;
  onSend: (value: string, attachments?: ChatAttachment[]) => Promise<void>;
}) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [fileError, setFileError] = useState("");
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  async function addFiles(files: FileList | null) {
    if (!files?.length) {
      return;
    }

    setFileError("");
    const room = MAX_IMAGES - attachments.length;
    if (room <= 0) {
      setFileError(`最多上传 ${MAX_IMAGES} 张病理图片。`);
      return;
    }

    const selected: File[] = [];
    const rejected: string[] = [];
    for (const file of Array.from(files)) {
      if (!file.type.startsWith("image/")) {
        rejected.push(file.name);
        continue;
      }
      if (file.size > MAX_IMAGE_BYTES) {
        rejected.push(file.name);
        continue;
      }
      if (selected.length < room) {
        selected.push(file);
      }
    }

    const next = await Promise.all(
      selected.map(async (file) => ({
        type: "image" as const,
        name: file.name,
        content_type: file.type || "image/*",
        url: await readFileAsDataUrl(file),
        size: file.size
      }))
    );

    setAttachments((prev) => [...prev, ...next].slice(0, MAX_IMAGES));
    if (rejected.length || selected.length < files.length) {
      setFileError("已忽略非图片、超过 4MB 或超出数量限制的文件。");
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function submit() {
    const nextValue =
      value.trim() || (attachments.length ? "请结合上传的病理图像进行 MDT 会诊。" : "");
    if (!nextValue) {
      return;
    }
    const outgoingAttachments = attachments;
    void onSend(nextValue, outgoingAttachments);
    setValue("");
    setAttachments([]);
    setFileError("");
  }

  const canSend = Boolean(value.trim() || attachments.length);

  return (
    <div className="panel rounded-lg p-2">
      {attachments.length > 0 ? (
        <div className="mb-2 flex flex-wrap gap-2 px-1">
          {attachments.map((attachment, index) => (
            <span
              className="inline-flex max-w-[210px] items-center gap-1 rounded-md border border-ocean/20 bg-ocean/5 px-2 py-1 text-xs text-[var(--color-ink)]"
              key={`${attachment.name}-${index}`}
            >
              <ImagePlus size={13} />
              <span className="truncate">{attachment.name}</span>
              <button
                className="rounded p-0.5 text-[var(--color-ink-soft)] hover:bg-white hover:text-[var(--color-ink)]"
                onClick={() =>
                  setAttachments((prev) => prev.filter((_, itemIndex) => itemIndex !== index))
                }
                type="button"
                title="移除图片"
              >
                <X size={12} />
              </button>
            </span>
          ))}
        </div>
      ) : null}
      <div className="flex items-end gap-2">
        <input
          accept="image/*"
          className="hidden"
          multiple
          onChange={(event) => void addFiles(event.target.files)}
          ref={fileInputRef}
          type="file"
        />
        <button
          className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg border border-[var(--color-line)] bg-white text-ocean hover:bg-ocean/5 disabled:cursor-not-allowed disabled:opacity-45"
          disabled={disabled}
          onClick={() => fileInputRef.current?.click()}
          title="上传病理图片"
          type="button"
        >
          <ImagePlus size={17} />
        </button>
        <textarea
          className="max-h-32 min-h-12 flex-1 resize-y rounded-lg border border-[var(--color-line)] bg-white/82 px-3 py-2 text-sm leading-6 outline-none focus:border-ocean/50 focus:bg-white"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault();
              submit();
            }
          }}
          placeholder="输入病例、分期、病理、基因检测和需要讨论的问题。Ctrl / Cmd + Enter 发送。"
          rows={2}
          value={value}
        />
        <button
          className="flex h-12 shrink-0 items-center gap-2 rounded-lg bg-ocean px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-ocean/45"
          disabled={disabled || !canSend}
          onClick={submit}
          type="button"
        >
          <SendHorizonal size={16} />
          发送
        </button>
      </div>
      {fileError ? (
        <p className="mt-1 px-1 text-[11px] leading-4 text-amber-700">{fileError}</p>
      ) : null}
      <p className="mt-1 px-1 text-[11px] leading-4 text-[var(--color-ink-soft)]">
        仅供科研和辅助讨论使用，具体诊疗请以临床医生判断为准。
      </p>
    </div>
  );
}
