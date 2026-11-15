const messages = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const newChatButton = document.getElementById("newChatButton");
const welcomeTemplate = document.getElementById("welcome").cloneNode(true);
const assistantName = welcomeTemplate.querySelector("h1").textContent.trim();
let waiting = false;

console.info("[Company RAG] markdown-v1 已加载");

// 所有动态内容均通过 DOM 节点构造；回答中的原始 HTML 不会执行。
function element(tag, className = "", text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function scrollToBottom() {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
}

function appendMessage(message) {
    messages.appendChild(message);
    scrollToBottom();
}

function assistantHeader() {
    return element("div", "assistant-header", assistantName);
}

// MathJax：同一页面只加载一次，主 CDN 失败时尝试备用 CDN。
let mathJaxLoadPromise = null;

function mathJaxReady() {
    return typeof window.MathJax?.typesetPromise === "function";
}

function ensureMathJax() {
    if (mathJaxReady()) return Promise.resolve(window.MathJax);
    if (mathJaxLoadPromise) return mathJaxLoadPromise;

    window.MathJax = {
        tex: {
            inlineMath: [["\\(", "\\)"], ["$", "$"]],
            displayMath: [["\\[", "\\]"], ["$$", "$$"]],
            processEscapes: true,
        },
        options: {
            skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
        },
        startup: { typeset: false },
    };

    const sources = [
        "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
        "https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
    ];
    mathJaxLoadPromise = new Promise((resolve, reject) => {
        function trySource(index) {
            if (index >= sources.length) {
                reject(new Error("MathJax CDN 加载失败"));
                return;
            }
            const script = element("script");
            script.src = sources[index];
            script.async = true;
            script.dataset.ragMathjax = "true";
            const next = () => {
                script.remove();
                trySource(index + 1);
            };
            script.addEventListener("load", () => {
                if (mathJaxReady()) resolve(window.MathJax);
                else next();
            });
            script.addEventListener("error", next);
            document.head.appendChild(script);
        }
        trySource(0);
    });
    return mathJaxLoadPromise;
}

async function typesetMath(node) {
    try {
        const mathJax = await ensureMathJax();
        await mathJax.typesetPromise([node]);
    } catch (error) {
        // 加载失败时保留 LaTeX 原文，聊天、表格和引用仍可使用。
        console.warn("[Company RAG] 公式渲染失败：", error);
    }
}

// 表格中兼容中文竖线与常见的横线字符。
function normalizeTableText(text) {
    return String(text)
        .replace(/[\u200B-\u200D\uFEFF]/g, "")
        .replaceAll("｜", "|")
        .replace(/[‐‑‒–—−]/g, "-");
}

function splitTableRow(line) {
    let row = normalizeTableText(line).trim();
    if (row.startsWith("|")) row = row.slice(1);
    if (row.endsWith("|")) row = row.slice(0, -1);
    return row.split("|").map(cell => cell.trim());
}

function isTableSeparator(line) {
    const cells = splitTableRow(line);
    return cells.length >= 2 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
}

function isPossibleTableRow(line) {
    const normalized = normalizeTableText(line).trim();
    return normalized.includes("|") && splitTableRow(normalized).length >= 2;
}

function safeUrl(value) {
    try {
        const url = new URL(value.trim(), window.location.href);
        return ["http:", "https:", "mailto:"].includes(url.protocol) ? url.href : null;
    } catch {
        return null;
    }
}

function renderInline(container, text) {
    let rest = text;
    while (rest) {
        let match;
        // 优先保留转义字符、行内代码和公式，避免其中的 *、_ 被当作强调。
        if ((match = rest.match(/^\\([\\`*_{}\[\]()#+.!>~|\-])/))) {
            container.append(match[1]);
        } else if ((match = rest.match(/^(`+)([\s\S]*?)\1(?!`)/))) {
            container.appendChild(element("code", "", match[2].replace(/\n/g, " ")));
        } else if ((match = rest.match(/^(\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$(?:[^$]|\$(?!\$))+\$\$|\$(?!\s)[^$\n]+?\$)/))) {
            container.append(match[0]);
        } else if ((match = rest.match(/^!\[([^\]]*)\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)/))) {
            const url = safeUrl(match[2] || match[3]);
            if (url && !url.startsWith("mailto:")) {
                const img = element("img");
                img.src = url;
                img.alt = match[1];
                img.loading = "lazy";
                container.appendChild(img);
            } else container.append(match[0]);
        } else if ((match = rest.match(/^\[([^\]]+)\]\((?:<([^>]+)>|([^\s)]+))(?:\s+"[^"]*")?\)/))) {
            const url = safeUrl(match[2] || match[3]);
            if (url) {
                const link = element("a");
                link.href = url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                renderInline(link, match[1]);
                container.appendChild(link);
            } else container.append(match[0]);
        } else if ((match = rest.match(/^(\*\*\*|___)(?=\S)([\s\S]*?\S)\1/))) {
            const strong = element("strong");
            const em = element("em");
            renderInline(em, match[2]);
            strong.appendChild(em);
            container.appendChild(strong);
        } else if ((match = rest.match(/^(\*\*|__)(?=\S)([\s\S]*?\S)\1/))) {
            const strong = element("strong");
            renderInline(strong, match[2]);
            container.appendChild(strong);
        } else if ((match = rest.match(/^~~(?=\S)([\s\S]*?\S)~~/))) {
            const del = element("del");
            renderInline(del, match[1]);
            container.appendChild(del);
        } else if ((match = rest.match(/^(\*|_)(?=\S)([^\n]*?\S)\1/))) {
            const em = element("em");
            renderInline(em, match[2]);
            container.appendChild(em);
        } else if ((match = rest.match(/^ {2,}\n/))) {
            container.appendChild(element("br"));
        } else if (rest[0] === "\n") {
            container.append(" ");
            match = ["\n"];
        } else {
            // 连续普通文本一次插入，长回答也无需逐字符扫描剩余内容。
            const next = rest.slice(1).search(/[\\`*!_~$\[\n ]/);
            match = [rest.slice(0, next < 0 ? rest.length : next + 1)];
            container.append(match[0]);
        }
        rest = rest.slice(match[0].length);
    }
}

function tableRow(values, tag, columnCount = values.length) {
    const row = element("tr");
    for (let column = 0; column < columnCount; column += 1) {
        const cell = element(tag);
        renderInline(cell, values[column] ?? "");
        row.appendChild(cell);
    }
    return row;
}

function appendTable(container, headerLine, bodyLines) {
    const headers = splitTableRow(headerLine);
    const wrapper = element("div", "rag-table-wrap");
    const table = element("table", "rag-table");
    const thead = element("thead");
    const tbody = element("tbody");
    thead.appendChild(tableRow(headers, "th"));
    for (const line of bodyLines) {
        const values = splitTableRow(line);
        // 少列补空，多列保留。
        tbody.appendChild(tableRow(values, "td", Math.max(headers.length, values.length)));
    }
    table.append(thead, tbody);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
}

function listMarker(line) {
    return line.match(/^(\s*)([-+*]|\d+[.)])\s+(.+)$/);
}

function isBlockStart(lines, index) {
    const line = lines[index];
    return /^ {0,3}(?:#{1,6}(?:\s|$)|(`{3,}|~{3,})|>|(?:-{3,}|\*{3,}|_{3,})\s*$|\$\$\s*$)/.test(line)
        || Boolean(listMarker(line))
        || (isPossibleTableRow(line) && isTableSeparator(lines[index + 1] || ""));
}

function renderBlocks(container, lines) {
    let index = 0;
    while (index < lines.length) {
        const line = lines[index];
        if (!line.trim()) { index += 1; continue; }

        const fence = line.match(/^ {0,3}(`{3,}|~{3,})([^`]*)$/);
        if (fence) {
            const content = [];
            const endFence = new RegExp(`^ {0,3}${fence[1][0]}{${fence[1].length},}\\s*$`);
            index += 1;
            while (index < lines.length && !endFence.test(lines[index])) content.push(lines[index++]);
            if (index < lines.length) index += 1;
            const pre = element("pre");
            pre.appendChild(element("code", "", content.join("\n")));
            container.appendChild(pre);
            continue;
        }
        if (/^ {0,3}\$\$\s*$/.test(line)) {
            const math = ["$$"];
            index += 1;
            while (index < lines.length && !/^ {0,3}\$\$\s*$/.test(lines[index])) math.push(lines[index++]);
            if (index < lines.length) { math.push("$$"); index += 1; }
            container.appendChild(element("div", "markdown-math", math.join("\n")));
            continue;
        }
        const heading = line.match(/^ {0,3}(#{1,6})(?:\s+|$)(.*?)\s*#*\s*$/);
        if (heading) {
            const node = element(`h${heading[1].length}`);
            renderInline(node, heading[2]);
            container.appendChild(node);
            index += 1;
            continue;
        }
        if (/^ {0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$/.test(line)) {
            container.appendChild(element("hr"));
            index += 1;
            continue;
        }
        if (/^ {0,3}>/.test(line)) {
            const quoteLines = [];
            while (index < lines.length && /^ {0,3}>/.test(lines[index])) {
                quoteLines.push(lines[index++].replace(/^ {0,3}> ?/, ""));
            }
            const quote = element("blockquote");
            renderBlocks(quote, quoteLines);
            container.appendChild(quote);
            continue;
        }
        if (isPossibleTableRow(line) && isTableSeparator(lines[index + 1] || "")) {
            const body = [];
            index += 2;
            while (index < lines.length && lines[index].trim() && isPossibleTableRow(lines[index])) {
                body.push(lines[index++]);
            }
            appendTable(container, line, body);
            continue;
        }
        const firstMarker = listMarker(line);
        if (firstMarker) {
            const indent = firstMarker[1].length;
            const ordered = /\d/.test(firstMarker[2][0]);
            const list = element(ordered ? "ol" : "ul");
            if (ordered) list.start = parseInt(firstMarker[2], 10);
            while (index < lines.length) {
                const marker = listMarker(lines[index]);
                if (!marker || marker[1].length !== indent || /\d/.test(marker[2][0]) !== ordered) break;
                index += 1;
                const item = element("li");
                let content = marker[3];
                const task = content.match(/^\[([ xX])\]\s+(.*)$/);
                if (task) {
                    const checkbox = element("input");
                    checkbox.type = "checkbox";
                    checkbox.checked = task[1].toLowerCase() === "x";
                    checkbox.disabled = true;
                    item.append(checkbox, " ");
                    content = task[2];
                }
                renderInline(item, content);
                const nested = [];
                while (index < lines.length && /^\s+\S/.test(lines[index]) &&
                    (lines[index].match(/^\s*/)[0].length > indent)) {
                    const nestedLine = lines[index++];
                    nested.push(nestedLine.slice(Math.min(indent + 2, nestedLine.match(/^\s*/)[0].length)));
                }
                if (nested.length) renderBlocks(item, nested);
                list.appendChild(item);
            }
            container.appendChild(list);
            continue;
        }
        const paragraph = [line];
        index += 1;
        while (index < lines.length && lines[index].trim() && !isBlockStart(lines, index)) {
            paragraph.push(lines[index++]);
        }
        const p = element("p");
        renderInline(p, paragraph.join("\n"));
        container.appendChild(p);
    }
}

function renderCitationButtons(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (!node.parentElement?.closest("code, pre, a, .markdown-math")) textNodes.push(node);
    }

    for (const node of textNodes) {
        const value = node.nodeValue || "";
        if (!/\[S\d+\]/.test(value)) continue;
        const fragment = document.createDocumentFragment();
        let lastIndex = 0;
        for (const match of value.matchAll(/\[S(\d+)\]/g)) {
            if (match.index > lastIndex) fragment.append(value.slice(lastIndex, match.index));
            const citation = `S${match[1]}`;
            const button = element("button", "citation-button", citation);
            button.type = "button";
            button.dataset.source = citation;
            button.title = `查看引用 ${citation}`;
            fragment.appendChild(button);
            lastIndex = match.index + match[0].length;
        }
        if (lastIndex < value.length) fragment.append(value.slice(lastIndex));
        node.replaceWith(fragment);
    }
}

function renderAnswer(container, text) {
    container.replaceChildren();
    renderBlocks(container, String(text).replace(/\r\n?/g, "\n").split("\n"));
    renderCitationButtons(container);
}

// 消息与证据卡片。
function addUserMessage(text) {
    document.getElementById("welcome")?.remove();
    const message = element("div", "message message-user");
    message.appendChild(element("div", "user-bubble", text));
    appendMessage(message);
}

function addLoading() {
    const message = element("div", "message");
    const row = element("div", "processing-row");
    const dots = element("div", "loading");
    dots.append(element("span"), element("span"), element("span"));
    const status = element("div", "processing-status", "正在准备…");
    row.append(dots, status);
    message.append(assistantHeader(), row);
    appendMessage(message);
    return { message, status };
}

function buildSources(sources) {
    if (!sources?.length) return null;
    const container = element("div", "sources");
    container.appendChild(element("div", "sources-title", "引用来源"));
    for (const source of sources) {
        const card = element("div", "source-card");
        card.dataset.source = source.citation_id;
        const version = source.version ? ` · ${source.version}` : "";
        card.append(
            element("div", "source-name", `[${source.citation_id}] ${source.file_name || "未知文件"}${version}`),
            element("div", "source-location", source.location_text || ""),
        );
        if (source.content) {
            const details = element("details");
            details.append(
                element("summary", "", "查看证据片段"),
                element("div", "source-content", source.content),
            );
            card.appendChild(details);
        }
        container.appendChild(card);
    }
    return container;
}

function addAssistantMessage(answer, sources) {
    const message = element("div", "message assistant-block");
    const content = element("div", "assistant-message");
    const answerNode = element("div", "assistant-answer");
    renderAnswer(answerNode, answer);
    content.appendChild(answerNode);
    const sourceNode = buildSources(sources);
    if (sourceNode) content.appendChild(sourceNode);
    message.append(assistantHeader(), content);
    messages.appendChild(message);
    typesetMath(answerNode);

    // 每条回答共用一个点击处理器；引用只定位到该回答的卡片。
    answerNode.addEventListener("click", event => {
        const button = event.target.closest(".citation-button");
        if (!button) return;
        const target = content.querySelector(`.source-card[data-source="${button.dataset.source}"]`);
        if (!target) return;
        const details = target.querySelector("details");
        if (details) details.open = true;
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("highlight");
        setTimeout(() => target.classList.remove("highlight"), 1600);
    });
    scrollToBottom();
}

// NDJSON 流：状态实时更新，答案等待 result 后一次性显示。
async function readEventStream(response, loading) {
    if (!response.body) throw new Error("浏览器未提供流式响应读取能力");
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let resultData = null;

    function handleLine(line) {
        const text = line.trim();
        if (!text) return;
        const event = JSON.parse(text);
        if (event.type === "status") {
            loading.status.textContent = event.message || "正在处理…";
            scrollToBottom();
        } else if (event.type === "result") {
            resultData = event.data || {};
        } else if (event.type === "error") {
            throw new Error(event.message || "后端处理失败");
        }
    }

    while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) handleLine(line);
        if (done) break;
    }
    if (buffer.trim()) handleLine(buffer);
    if (!resultData) throw new Error("后端没有返回最终结果");
    return resultData;
}

async function sendQuestion(question) {
    if (waiting) return;
    waiting = true;
    sendButton.disabled = true;
    addUserMessage(question);
    const loading = addLoading();
    try {
        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        if (!response.ok) {
            const raw = await response.text();
            let detail = raw;
            try {
                detail = JSON.parse(raw).detail || raw;
            } catch {
                // 非 JSON 响应继续使用原文。
            }
            throw new Error(detail || `HTTP ${response.status}`);
        }
        const data = await readEventStream(response, loading);
        loading.message.remove();
        addAssistantMessage(data.answer || "没有返回内容.", data.sources || []);
    } catch (error) {
        loading.message.remove();
        addAssistantMessage(`发生错误：${error.message || String(error)}`, []);
    } finally {
        waiting = false;
        sendButton.disabled = false;
        input.focus();
    }
}

function resizeInput() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

form.addEventListener("submit", event => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    resizeInput();
    sendQuestion(question);
});

input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});
input.addEventListener("input", resizeInput);
newChatButton.addEventListener("click", () => {
    messages.replaceChildren(welcomeTemplate.cloneNode(true));
    input.focus();
});

// 页面存活连接：20 秒 ping，断线 1 秒后重连，pagehide 时关闭。
let browserSessionSocket = null;
let browserSessionPingTimer = null;
let browserSessionReconnectTimer = null;
let pageIsClosing = false;

function connectBrowserSession() {
    if (pageIsClosing) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    browserSessionSocket = new WebSocket(`${protocol}//${window.location.host}/api/browser-session`);
    browserSessionSocket.addEventListener("open", () => {
        clearTimeout(browserSessionReconnectTimer);
        browserSessionReconnectTimer = null;
        clearInterval(browserSessionPingTimer);
        browserSessionPingTimer = setInterval(() => {
            if (browserSessionSocket?.readyState === WebSocket.OPEN) {
                browserSessionSocket.send("ping");
            }
        }, 20000);
    });
    browserSessionSocket.addEventListener("close", () => {
        clearInterval(browserSessionPingTimer);
        browserSessionPingTimer = null;
        if (!pageIsClosing) {
            browserSessionReconnectTimer = setTimeout(connectBrowserSession, 1000);
        }
    });
    browserSessionSocket.addEventListener("error", () => {});
}

connectBrowserSession();
window.addEventListener("pagehide", () => {
    pageIsClosing = true;
    clearTimeout(browserSessionReconnectTimer);
    browserSessionReconnectTimer = null;
    clearInterval(browserSessionPingTimer);
    browserSessionPingTimer = null;
    if (browserSessionSocket && browserSessionSocket.readyState <= WebSocket.OPEN) {
        browserSessionSocket.close();
    }
});
