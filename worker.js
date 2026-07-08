export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = 'https://pte-speaking-public.onrender.com';
    const targetUrl = target + url.pathname + url.search;

    const modified = new Request(targetUrl, {
      method: request.method,
      headers: new Headers(request.headers),
      body: request.body,
      redirect: 'manual',
    });
    modified.headers.set('Host', 'pte-speaking-public.onrender.com');

    let response = await fetch(modified);
    return response;
  }
};
