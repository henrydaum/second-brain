# Frontend Adapter Notes

`FrontendRuntime` owns shared product behavior: agent lifecycle, busy-state, approvals, history actions, tool status, conversation persistence, and chat pushes.

`BasePlatformAdapter` implementations should stay transport-focused:
- define `name` and `capabilities`
- create stable `FrontendSession` values for each user/chat/thread
- start and stop the transport
- render `FrontendAction` back to the platform
- normalize inbound transport input into `FrontendEvent` and pass it to `FrontendRuntime.handle_frontend_event(...)`

Keep platform-specific menus, callback token formats, and media APIs in the adapter package. Keep shared command/chat/approval/history behavior in the runtime.
