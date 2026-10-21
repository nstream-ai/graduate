# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately by email to
**shivanshu.dubey12@gmail.com**, with "Graduate security" in the subject.

Include what you found, how to reproduce it, and the version or commit. We aim to acknowledge a
report within 5 working days and to agree on a fix and disclosure date with you.

## Scope

Graduate runs locally on traces you provide and sends requests only to the model endpoint you
configure (`--base-url`); API keys are read from an environment variable you name and are never
written to the report. Problems in how Graduate handles traces, keys or report output are in
scope. Problems in the model servers or providers you point it at are not.

## Supported versions

Only the latest release receives fixes during the research preview.
