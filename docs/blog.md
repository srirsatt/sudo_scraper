# Internship links without Instagram

*Draft for review.*

I wanted to stay off Instagram without missing the internship postings zero2sudo
shares. So I built a small tool that brings those links to Telegram.

It runs on my base M4 Pro Mac mini: 24 GB of memory, 512 GB of storage. Qwen3-VL
runs locally through Ollama. The collector saves stories and their original links,
then the classifier reads the linked job description. If that page isn't usable,
it looks at the image or sampled video frames.

The filter is deliberately inclusive. Anything involving coding or development
can qualify, including ML, robotics, automation, and hybrid roles. Requiring
“Software Engineer” in the title turned out to exclude useful internships.

Matches arrive in Telegram. Held stories come with their saved media and two
buttons: Approve or Skip. My decision is stored separately from the model's output,
and approved links use the same delivery tracker as automatic matches.

The useful engineering was mostly around the model: preserving real application
URLs, caching work, keeping review decisions, and handling a timeout without blindly
sending the same alert again. I also caught a library setting that meant “sleep
before each request,” rather than “network timeout.” That explained a very slow run.

It's an unofficial personal tool, with the limits that implies. Stories can be
missed, sampled frames leave gaps, and Instagram can restrict access. It gives me
an inbox for the opportunities I care about without opening the feed to find them.

[Source](https://github.com/srirsatt/sudo_scraper)
