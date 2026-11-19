// The GenAI page: words of the letter tiles, from the language model on the board's Ara240.
//
// LOCATE TILES is the Tiles page's: the arm parks, the snapshot is taken, the tiles are found and their letters read.
// The letters on the tiles ready to pick up are what a word is made of. Down the side are the words of the model's
// library that those letters spell (applib/genai/library.py), by length, the longest first: there at once, since the
// library does not depend on the tiles. LONGEST WORD and 3, 4 and 5 LETTERS set one of them down, picked at random
// of its length; a click on a word sets that one down. A question typed is put to the model, which answers in one
// word: the arm spells it if the tiles can, or another word of the list that the model holds right too; what it said
// and why not is shown under the question. The model takes minutes to load when the app starts: until then the list
// and its buttons work, and ASK waits.
// Setting a word down is PLACE, as on the Tiles page, with the grip height and the reader chosen there; the tiles are
// found again after it, and the list follows the letters left.

import { $, api, fmt, mountHeader, report, setText, showHeader, startPolling } from '/common/page.js';
import { PictureLayer } from '/common/layer.js';

const FEED = '/api/camera/feed';
const FEED_RETRY_MS = 3000;
const FEED_NOTES = { idle: 'starting the camera…', starting: 'starting the camera…' };
const FEED_BADGES = { live: '● LIVE', failed: 'NO CAMERA' };
const STATUS_WORDS = {
  ready: 'ready to pick up', 'too close': 'too close', elsewhere: 'outside the pickup area',
  'out of reach': 'out of reach',
};
const LENGTH_NAMES = { 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine' };

mountHeader('genai');
let state = null;
let inFlight = 0;
let isLost = false;
let asked = '';                 // the question on its way to the model
let building = '';              // the word a click or a pick is setting down
let wordsShown = '';

const scene = new PictureLayer($('scene'), (width, scale) => `/api/tiles/scene.png?w=${width}&s=${scale}&areas=1`);
const tiles = () => (state ? state.tiles : null);
const found = () => (tiles() ? tiles().found : null);
const placing = () => (tiles() ? tiles().word : null);
const genai = () => (state ? state.genai : null);
const words = () => (genai() ? genai().words : []);
const isBusy = () => inFlight > 0 || !state || Boolean(state.doing) || state.queued > 0;
// The tiles PLACE has started on, by number: no longer, or perhaps not, where the snapshot has them.
const moved = () => (placing() ? placing().steps.filter((step) => step.status !== 'waiting').length : 0);
const canBuild = () => !isBusy() && Boolean(found()) && moved() === 0 && !tiles().problem;

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
  report(reply, reply.word ? `${what} ${reply.word}` : what);
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

const locate = () => command('tiles/locate', {}, 'LOCATE TILES');

async function build(body, shown) {
  if (!canBuild()) return;
  building = shown;
  await command('genai/build', body, 'BUILD');
  building = '';
  render();
}

async function ask() {
  const question = $('question').value.trim().replace(/\s+/g, ' ');
  if (!question || $('ask').disabled) return;
  asked = question;
  await command('genai/ask', { question }, 'ASK');
  asked = '';
  render();
}

// --- drawing it all --------------------------------------------------------------------------------

function render() {
  const word = placing();
  const isShown = tiles() && !tiles().problem && !tiles().view_problem;
  scene.show(isShown ? JSON.stringify([found() ? found().stamp : null,
    word ? [word.status, word.steps.map((step) => step.status)] : null]) : null);
  if (!state) return;
  renderPage();
  renderPicture();
  renderPicks();
  renderAsk();
  renderLibrary();
}

function renderPage() {
  const { problem } = tiles();
  $('blocked').hidden = !problem;
  document.querySelector('main.genai').classList.toggle('is-blocked', Boolean(problem));
  $('blocked-why').textContent = problem ? `${problem[0].toUpperCase()}${problem.slice(1)}.` : '';
  $('locate').disabled = Boolean(problem) || isBusy();
  const result = found();
  $('locate-line').innerHTML = result ? locateLine(result)
    : '<span class="note">Set the tiles down in the pickup area, then LOCATE TILES: the arm moves out of the picture '
      + 'and the tiles are found and read in a snapshot.</span>';
  $('word-line').innerHTML = wordLine(result, placing());
}

function locateLine(result) {
  const parts = Object.entries(STATUS_WORDS).filter(([status]) => result.counts[status])
    .map(([status, said]) => `<b>${result.counts[status]}</b> ${said}`);
  if (!result.tiles.length) return 'No tiles found: are they in view, the arm out of the picture?';
  const offer = result.on_offer ? `letters to build with: <b class="letters">${result.on_offer}</b>`
    : 'no letters on tiles ready to pick up';
  return `<b>${result.tiles.length}</b> tiles: ${parts.join(', ')}; ${offer} <span class="note">· the snapshot of `
    + `${result.taken}</span>`;
}

// How the word being set down is going, or how the last one went.
function wordLine(result, word) {
  if (!result) return '';
  if (!word) return result.placed ? `${escaped(result.placed)}; the tiles found again.` : '';
  const head = `<b class="letters">${word.letters}</b>: `;
  const placed = word.steps.filter((step) => step.status === 'placed').length;
  if (word.status === 'placing') {
    const carrying = word.steps.find((step) => step.status === 'carrying');
    return `${head}${placed} of ${word.steps.length} set down${carrying ? `, carrying ${carrying.letter}` : ''}`;
  }
  const again = ' <span class="note">· LOCATE TILES again before the next word</span>';
  if (word.status === 'done') return `${head}all ${word.steps.length} set down${again}`;
  return `${head}${word.status} with ${placed} of ${word.steps.length} set down: ${escaped(word.detail)}${again}`;
}

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
function escaped(text) {
  return String(text).replace(/[&<>"]/g, (character) => ESCAPES[character]);
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

// Each button says how many words it has to pick from; with none, it is off.
function renderPicks() {
  const list = words();
  const longest = list.length ? list[0].length : 0;
  for (const button of $('picks').children) {
    const { pick } = button.dataset;
    const count = list.filter((word) => word.length === (pick === 'longest' ? longest : Number(pick))).length;
    const label = pick === 'longest' ? 'LONGEST WORD' : `${pick} LETTERS`;
    const note = !found() ? 'locate the tiles first' : !count ? 'none in the list'
      : pick === 'longest' ? `${longest} letters · ${count} to pick from` : `${count} to pick from`;
    const markup = `${label}<small>${note}</small>`;
    if (button.innerHTML !== markup) button.innerHTML = markup;
    button.disabled = !canBuild() || !count;
  }
}

function renderAsk() {
  const { model, answer } = genai();
  const isReady = model.status === 'ready';
  $('question').disabled = !isReady;
  $('ask').disabled = !isReady || !canBuild() || !genai().letters || !$('question').value.trim();
  const isAnswered = answer && (!asked || (answer.question === asked && !model.asking));
  const question = asked || (answer ? answer.question : '');
  $('answer-line').innerHTML = !question ? '' : `<span class="asked">${escaped(question)}</span><br>`
    + (isAnswered ? answerLine(answer) : '<span class="thinking">the model is answering…</span>');
}

// What the model said, what the arm spells of it, and why they differ if they do.
function answerLine(answer) {
  const said = answer.said ? `<b>${escaped(answer.said)}</b>` : 'nothing';
  if (answer.word && answer.word === answer.said) return `<b class="spelled">${answer.word}</b>${placedNote(answer)}`;
  if (answer.word) {
    return `The model says ${said}: ${escaped(answer.detail.split(';')[0])}. <b class="spelled">${answer.word}</b> is `
      + `right too${placedNote(answer)}`;
  }
  return `The model says ${said}: ${escaped(answer.detail)}. Nothing set down.`;
}

function placedNote(answer) {
  const { placed } = answer;
  if (!placed) return ' <span class="note">· setting it down</span>';
  if (placed.status === 'done') return ' <span class="note">· set down</span>';
  return ` <span class="note">· not set down: ${escaped(placed.detail || placed.status)}</span>`;
}

function renderLibrary() {
  const { model, letters } = genai();
  renderModel(model);
  const list = words();
  setText($('word-count'), list.length ? `${list.length} words` : '');
  setText($('library-line'), !model.library ? 'The list is made by the language model, once, when it has loaded: it '
    + 'goes through the words it knows and marks the everyday English ones. After that it is here at once.'
    : !found() ? `The model's library has ${model.library} everyday English words. LOCATE TILES, and those the tiles `
      + 'spell are listed here.'
      : `Of the ${model.library} everyday English words the model marked, those the letters ${letters || '(none)'} `
        + 'spell, each tile once. Click one to set it down.');
  const shown = JSON.stringify([list, canBuild(), building]);
  if (shown === wordsShown) return;
  wordsShown = shown;
  const lengths = [...new Set(list.map((word) => word.length))];
  $('words').innerHTML = lengths.map((length) => {
    const group = list.filter((word) => word.length === length).sort();
    return `<div><h3>${LENGTH_NAMES[length] || length} letters <b>· ${group.length}</b></h3><div class="group">`
      + group.map((word) => `<button class="word${word === building ? ' building' : ''}" data-word="${word}" `
        + `${canBuild() ? '' : 'disabled'}>${word}</button>`).join('') + '</div></div>';
  }).join('') || (found() && model.library ? '<span class="none">No word of the library can be spelled with these '
    + 'letters.</span>' : '');
}

// The model's pill: loading, with how long it has been and usually takes; ready; or why not.
function renderModel(model) {
  const pill = $('model');
  const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  const said = {
    idle: 'language model: starting',
    waiting: 'waiting for the Ara240',
    loading: `loading onto the Ara240 · ${clock(model.seconds)} of about ${clock(model.typical || 0)}`,
    marking: `marking its words · ${fmt(model.seconds, 0)} s of about ${model.typical || 0}`,
    ready: model.asking ? 'answering…' : `${model.name} on the Ara240 · ready`,
    failed: 'language model not available',
  }[model.status] || model.status;
  pill.className = `pill ${{ ready: model.asking ? 'busy' : 'ready', failed: 'failed' }[model.status] || 'connecting'}`;
  pill.title = model.detail || `${model.name}, on the Ara240`;
  setText(pill.lastElementChild, said);
}

// The live view only while the page is in sight.
function connectFeed() {
  if (!document.hidden) $('feed').src = `${FEED}?t=${Date.now()}`;
  else $('feed').removeAttribute('src');
}

// --- wiring ----------------------------------------------------------------------------------------

$('locate').addEventListener('click', locate);
$('picks').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-pick]');
  if (button && !button.disabled) build({ pick: button.dataset.pick }, '');
});
$('words').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-word]');
  if (button && !button.disabled) build({ word: button.dataset.word }, button.dataset.word);
});
$('question').addEventListener('input', render);
$('question').addEventListener('keydown', (event) => { if (event.key === 'Enter') ask(); });
$('ask').addEventListener('click', ask);
$('feed').addEventListener('error', () => { if (!document.hidden) setTimeout(connectFeed, FEED_RETRY_MS); });
document.addEventListener('visibilitychange', connectFeed);
connectFeed();
render();
startPolling((next) => show(next));
