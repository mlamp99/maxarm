// The Chat page: a request to the language model on the board's Ara240 -- "go", "a 7-letter word", "an animal",
// "elaborate" -- which guesses words, has them checked against the tiles in code, and has the arm set the best down.
//
// On the left, the desk: the live view with the tiles drawn over it, as on the Tiles page; the letters on the tiles
// ready to pick up, as tiles; and the word being set down, tile by tile as the arm carries each. LOCATE TILES is the
// Tiles page's, and so are the tile offset and the reader the arm sets words down for.
// On the right, the conversation: each request, the model's tool calls as they happen -- its guesses, green those
// that fit, the one chosen filled -- and its answer, with what it took: seconds, model calls, tokens read and written.
// A word set down, or not, is said when the arm is done. The model is told what happened before in a line, never the
// conversation itself: NEW VISITOR starts both again. The model takes minutes to load after the board starts; its pill
// says how far it has got, and the box waits for it. One request at a time: STOP cuts it short; RESET stops the arm.

import { $, api, fmt, mountHeader, report, setText, showHeader, startPolling } from '/common/page.js';
import { PictureLayer } from '/common/layer.js';

const FEED = '/api/camera/feed';
const FEED_RETRY_MS = 3000;
const FEED_NOTES = { idle: 'starting the camera…', starting: 'starting the camera…' };
const FEED_BADGES = { live: '● LIVE', failed: 'NO CAMERA' };
const SUGGESTIONS = ['Go!', 'A 7-letter word', 'A large farm animal', 'Elaborate',
  'Look at the tiles again, then a 5-letter word', "A word, but don't place it"];
const MODEL_WORDS = {
  ready: 'ready', loading: 'loading', waiting: 'waiting for the Ara240', stopped: 'not running',
  'not installed': 'not installed', failed: 'not answering', checking: 'looking for it', off: 'off',
};
const MODEL_PILLS = { ready: 'ready', loading: 'connecting', waiting: 'connecting', checking: 'connecting', failed: 'failed' };
const TOOL_ICONS = { find_word: '⌕', detect_tiles: '◎' };
const STEP_WORDS = { carrying: 'carrying', placed: 'set down', waiting: 'next' };

mountHeader('chat');
let state = null;
let inFlight = 0;
let isLost = false;
let messagesShown = '';
let isPinned = true;            // the conversation scrolled to its end: kept there as it grows

const scene = new PictureLayer($('scene'), (width, scale) => `/api/tiles/scene.png?w=${width}&s=${scale}&areas=1`);
const tiles = () => (state ? state.tiles : null);
const found = () => (tiles() ? tiles().found : null);
const placing = () => (tiles() ? tiles().word : null);
const chat = () => (state ? state.chat : null);
const isModelReady = () => Boolean(chat()) && chat().model.state === 'ready';
const isThinking = () => Boolean(chat() && chat().thinking);
const isArmBusy = () => inFlight > 0 || !state || Boolean(state.doing) || state.queued > 0;

// --- talking to the app ------------------------------------------------------------------------------

async function command(path, body, what) {
  inFlight += 1;
  render();
  let reply;
  try {
    reply = await api(path, body);
  } catch {
    reply = { status: 'error', detail: 'the calibration app is not answering' };
  }
  inFlight -= 1;
  report(reply, what);
  show(reply.state || state);
  return reply;
}

function show(next) {
  if (next && state && next.stamp < state.stamp) return;        // a poll answered before a newer reply
  if (next && isLost) connectFeed();                            // the app is back: its live view is a new one
  isLost = !next;
  state = next;
  showHeader(state);
  render();
}

async function ask(text) {
  const request = text.trim().replace(/\s+/g, ' ');
  if (!request || !isModelReady() || isThinking() || inFlight > 0) return;
  $('text').value = '';
  isPinned = true;
  await command('chat/ask', { text: request }, 'ASK');
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  const isShown = tiles() && !tiles().problem && !tiles().view_problem;
  const word = placing();
  scene.show(isShown ? JSON.stringify([found() ? found().stamp : null,
    word ? [word.status, word.steps.map((step) => step.status)] : null]) : null);
  if (!state) return;
  renderPage();
  renderPicture();
  renderDesk();
  renderModel();
  renderMessages();
  renderAsk();
}

function renderPage() {
  const { problem } = tiles();
  $('blocked').hidden = !problem;
  document.querySelector('main.chat').classList.toggle('is-blocked', Boolean(problem));
  $('blocked-why').textContent = problem ? `${problem[0].toUpperCase()}${problem.slice(1)}.` : '';
  $('locate').disabled = Boolean(problem) || isArmBusy() || Boolean(chat().placing);
}

function renderPicture() {
  const { frame_size: size, view_problem: viewProblem } = tiles();
  if (size) $('feed-box').style.aspectRatio = `${size[0]} / ${size[1]}`;
  const { feed } = state.camera;
  $('feed-badge').textContent = FEED_BADGES[feed.status] || '';
  $('feed-badge').className = `feed-badge ${feed.status}`;
  $('feed-note').textContent = viewProblem || (feed.status === 'failed' ? feed.detail : FEED_NOTES[feed.status] || '');
  $('feed-device').textContent = feed.status === 'live' ? feed.detail : '';
}

// The letters to use, as tiles; the word being set down, a tile for each letter as the arm gets on with it.
function renderDesk() {
  const result = found();
  const offer = result ? result.on_offer : '';
  const offerKey = `offer:${offer}:${Boolean(result)}`;
  if ($('on-offer').dataset.key !== offerKey) {
    $('on-offer').dataset.key = offerKey;
    $('on-offer').innerHTML = offer ? [...offer].map((letter) => `<span class="tile">${letter}</span>`).join('')
      : `<span class="hint">${result ? 'none: no tile ready to pick up has its letter read' : 'not looked at yet'}</span>`;
  }
  const word = placing();
  $('word-row').hidden = !word;
  if (word) {
    const placed = word.steps.filter((step) => step.status === 'placed').length;
    setText($('word-label'), word.status === 'placing' ? 'setting down' : word.status === 'done' ? 'set down'
      : 'not finished');
    const wordKey = JSON.stringify([word.letters, word.steps.map((step) => step.status)]);
    if ($('word-tiles').dataset.key !== wordKey) {
      $('word-tiles').dataset.key = wordKey;
      $('word-tiles').innerHTML = word.steps.map((step) => {
        const kind = step.status === 'placed' ? 'placed' : step.status === 'carrying' ? 'carrying'
          : step.status === 'waiting' ? 'slot' : 'failed';
        return `<span class="tile ${kind}" title="${step.letter}: ${STEP_WORDS[step.status] || step.status}">`
          + `${step.letter}</span>`;
      }).join('');
    }
    setText($('word-note'), word.status === 'placing' ? `${placed} of ${word.steps.length}` : word.detail || '');
  }
  setText($('desk-note'), !result
    ? 'LOCATE TILES, or ask the model to look: the arm moves out of the picture and the camera reads the tiles.'
    : `${result.tiles.length} tiles in the snapshot of ${result.taken}; the tile offset and the reader are the Tiles `
      + 'page\'s.');
}

function renderModel() {
  const { model } = chat();
  const pill = $('model');
  let said = `${model.name} · ${MODEL_WORDS[model.state] || model.state}`;
  if (model.state === 'loading' && model.elapsed_s !== null) {
    said = `loading onto the Ara240 · ${clock(model.elapsed_s)} of about ${clock(model.typical_s)}`;
  }
  if (isThinking()) said = `${model.name} · answering`;
  pill.className = `pill ${isThinking() ? 'busy' : MODEL_PILLS[model.state] || 'off'}`;
  pill.title = model.detail || model.name;
  setText(pill.lastElementChild, said);
}

function renderMessages() {
  const { messages, thinking } = chat();
  const elapsed = thinking ? thinking.seconds : 0;
  const shown = JSON.stringify([messages.map((message) => [message.n, message.is_done, message.text.length]),
    thinking ? [thinking.doing, thinking.calls, elapsed] : null]);
  if (shown === messagesShown) return;
  messagesShown = shown;
  const box = $('messages');
  let fitting = new Set();                     // the words the model found that fit, since the last request
  const parts = messages.map((message) => {
    if (message.role === 'user') fitting = new Set();
    if (message.role === 'step' && message.fits) {
      message.fits.filter((fit) => fit.is_fit).forEach((fit) => fitting.add(fit.word));
    }
    if (message.role === 'event' && message.word) fitting.add(message.word);
    return messageHtml(message, fitting);
  });
  if (thinking) {
    parts.push(`<div class="bubble assistant thinking">${escaped(thinking.doing)}<span class="dots"><span>.</span>`
      + `<span>.</span><span>.</span></span><span class="meta">${elapsed} s · model call ${thinking.calls}`
      + `${thinking.steps ? ` · tool call ${thinking.steps}` : ''}</span></div>`);
  }
  if (!parts.length) {
    parts.push('<div class="empty">Ask for a word. The model guesses words of the tiles, the code checks each guess '
      + 'against them -- the model cannot count letters -- and the arm sets the best one down. Try <b>go</b>, '
      + '<b>a 7-letter word</b>, a clue like <b>a farm animal</b>, then <b>elaborate</b>.</div>');
  }
  box.innerHTML = parts.join('');
  if (isPinned) box.scrollTop = box.scrollHeight;
}

function messageHtml(message, fitting) {
  if (message.role === 'user') return `<div class="bubble user">${escaped(message.text)}</div>`;
  if (message.role === 'event') {
    return `<div class="event ${message.is_ok ? '' : 'failed'}">${message.is_ok ? '✓' : '✕'} ${escaped(message.text)}`
      + '</div>';
  }
  if (message.role === 'step') return stepHtml(message);
  const meta = message.is_failed ? `${fmt(message.seconds, 0)} s`
    : `${fmt(message.seconds, 0)} s · ${message.calls} model call${message.calls === 1 ? '' : 's'} · `
      + `${message.tokens_in} tokens read, ${message.tokens_out} written${message.is_cut ? ' · cut short' : ''}`;
  return `<div class="bubble assistant ${message.is_failed ? 'failed' : ''}">${withTiles(message.text, fitting)}`
    + `<span class="meta">${meta}</span></div>`;
}

// A tool call: its guesses as chips once checked -- green those that fit, the one set down filled, struck through
// those whose letters are not on the tiles, and those the judge says are not words -- and what became of it.
function stepHtml(message) {
  const classes = `step ${message.is_done ? '' : 'running'} ${message.is_failed ? 'failed' : ''}`;
  const head = `<span class="icon">${TOOL_ICONS[message.tool] || '•'}</span><span class="name">${message.tool}`
    + `${message.place === false ? ' · not to be set down' : ''}${message.is_done ? ` · ${fmt(message.seconds, 1)} s` : ''}`
    + '</span>';
  let result;
  if (!message.is_done) {
    result = escaped(message.argument || 'working…');
  } else if (message.fits && message.fits.length) {
    result = message.fits.map((fit) => guessChip(fit, message.word)).join('') + `<br>${verdict(message)}`;
  } else {
    result = escaped(message.text);
  }
  return `<div class="${classes}" title="${escaped(message.text)}">${head}<span class="result">${result}</span></div>`;
}

function guessChip(fit, chosen) {
  const blocked = !fit.is_fit && !fit.missing && fit.problem;
  const madeUp = !fit.is_fit && !fit.missing && !fit.problem && fit.is_word === false;
  const kind = fit.word === chosen ? 'fit chosen' : fit.is_fit ? 'fit' : blocked ? 'blocked' : madeUp ? 'made-up' : '';
  const title = fit.is_fit ? `fits: ${fit.word.length} letters` : blocked ? fit.problem
    : madeUp ? 'its letters are there, but it is not a word, the judge says'
      : `the tiles lack ${[...fit.missing].join(' ')}`;
  return `<span class="guess ${kind}" title="${escaped(title)}">${escaped(fit.word)}`
    + `${fit.is_fit ? ` <small>${fit.word.length}</small>` : ''}</span>`;
}

function verdict(message) {
  if (message.is_placed) return `→ the arm sets down ${tileWord(message.word)}`;
  if (message.word) return `→ best: <b>${escaped(message.word)}</b>, not set down${message.place === false ? ', as asked' : ''}`;
  return 'no word of them fits: the model is asked for others';
}

function renderAsk() {
  const { model } = chat();
  const isReady = isModelReady();
  $('text').disabled = !isReady;
  $('text').placeholder = isReady ? 'go, a 7-letter word, an animal…' : 'the language model is not ready';
  $('ask').disabled = !isReady || isThinking() || inFlight > 0;
  $('stop').hidden = !isThinking();
  $('forget').disabled = isThinking() || !chat().messages.length;
  for (const button of $('suggestions').children) button.disabled = !isReady || isThinking() || inFlight > 0;
  setText($('talk-note'), isReady ? (chat().placing ? `The arm is setting down ${chat().placing}: words can be `
    + 'checked meanwhile, not set down.' : '') : unreadyNote(model));
}

function unreadyNote(model) {
  if (model.state === 'not installed') {
    return 'The language model is not installed on this board: connector/install.sh, with an Ara240 (11 GB).';
  }
  if (model.state === 'stopped') {
    return 'It is installed, not running: connector/run.sh, or connector/install-service.sh for every boot.';
  }
  if (model.state === 'loading') return 'The model loads onto the Ara240 for about 6 minutes after the connector starts.';
  return model.detail ? `${model.detail[0].toUpperCase()}${model.detail.slice(1)}.` : '';
}

// --- helpers -----------------------------------------------------------------------------------------

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
function escaped(text) {
  return String(text).replace(/[&<>"]/g, (character) => ESCAPES[character]);
}

function tileWord(word) {
  return `<span class="word-tiles">${[...word].map((letter) => `<span class="tile small">${letter}</span>`).join('')}`
    + '</span>';
}

// The model's answer, the words it found that fit drawn as tiles, whatever case it wrote them in.
function withTiles(text, words) {
  return String(text).split(/([A-Za-z]+)/).map((part, index) => (index % 2 === 1 && part.length > 1
    && words.has(part.toUpperCase()) ? tileWord(part.toUpperCase()) : escaped(part))).join('');
}

function clock(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

// The live view only while the page is in sight.
function connectFeed() {
  if (!document.hidden) $('feed').src = `${FEED}?t=${Date.now()}`;
  else $('feed').removeAttribute('src');
}

// --- wiring ----------------------------------------------------------------------------------------

$('suggestions').innerHTML = SUGGESTIONS.map((text) => `<button type="button">${escaped(text)}</button>`).join('');
$('suggestions').addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (button && !button.disabled) ask(button.textContent);
});
$('ask-form').addEventListener('submit', (event) => {
  event.preventDefault();
  ask($('text').value);
});
$('stop').addEventListener('click', () => command('chat/stop', {}, 'STOP'));
$('forget').addEventListener('click', () => command('chat/forget', {}, 'NEW VISITOR'));
$('locate').addEventListener('click', () => command('tiles/locate', {}, 'LOCATE TILES'));
$('messages').addEventListener('scroll', () => {
  const box = $('messages');
  isPinned = box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
});
$('feed').addEventListener('error', () => { if (!document.hidden) setTimeout(connectFeed, FEED_RETRY_MS); });
document.addEventListener('visibilitychange', connectFeed);
connectFeed();
render();
startPolling((next) => show(next));
