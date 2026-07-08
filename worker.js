export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = 'https://ztt625-tonigjw.hf.space';
    const targetUrl = target + url.pathname + url.search;

    const modified = new Request(targetUrl, {
      method: request.method,
      headers: new Headers(request.headers),
      body: request.body,
      redirect: 'manual',
    });
    modified.headers.set('Host', 'ztt625-tonigjw.hf.space');

    let response = await fetch(modified);
    return response;
  }
};
