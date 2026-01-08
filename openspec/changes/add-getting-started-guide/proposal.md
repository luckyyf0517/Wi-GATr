# Change: Add Getting Started Guide for First-Time Users

## Why

First-time users who have downloaded the Wi3R/WiPTR datasets need clear, step-by-step instructions on:
1. Where to place the downloaded data
2. How to verify the data structure is correct
3. How to run a quick test overfit to validate the setup
4. How to proceed to full training

The current README assumes users know exactly where to place data and provides multiple setup options (uv/Docker) without a clear recommended path for beginners.

## What Changes

- Add a comprehensive `GETTING_STARTED.md` guide with step-by-step instructions
- Include data directory structure verification steps
- Provide expected file listings for data validation
- Add troubleshooting section for common data placement issues
- Include example commands with exact paths
- Add environment setup validation commands

## Impact

- Affected specs: `getting-started` (new capability)
- Affected code: Documentation only (new file `GETTING_STARTED.md`)
- No breaking changes to existing code
- Enhances user onboarding experience
