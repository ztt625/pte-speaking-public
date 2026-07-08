export default {
  async fetch(request) {
    const url = new URL(request.url);
    // 代理到 HF Space 实际运行的地址
    const target = 'https://ztt625-tonigjw.hf.space';
    const targetUrl = target + url.pathname + url.search;

    const modified = new Request(targetUrl, {
      method: request.method,
      headers: new Headers(request.headers),
      body: request.body,
      redirect: 'manual',
    });
    modified.headers.set('Host', 'ztt625-tonigjw.hf.space');
    modified.headers.set('Origin', target);

    let response = await fetch(modified);

    // HTML/JS/CSS 里把 hf 的域名替换成你的域名
    const contentType = response.headers.get('content-type') || '';
    if (/text\/html|javascript|css/.test(contentType)) {
      let text = await response.text();
      text = text.replace(/ztt625-tonigjw\.hf\.space/g, 'pte.xiaohuni.com');
      text = text.replace(/huggingface\.co/g, 'pte.xiaohuni.com');
      response = new Response(text, {
        status: response.status,
        headers: response.headers,
      });
    }

    return response;
  }
};
