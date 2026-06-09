---
title: context.schema
sidebar_label: schema
---

Pydantic schemas for the Subvocal user context.

Defines the structure for contacts, calendar events, location, conversation,
and app state (including visible interactive elements).

## Classes

### `class Contact(BaseModel)`

Represents a user contact contact.

### `class CalendarEvent(BaseModel)`

Represents a calendar entry.

### `class LocationInfo(BaseModel)`

Represents the user's current spatial coordinates and address.

### `class Message(BaseModel)`

Represents a conversation turn in user-agent history.

### `class UIElement(BaseModel)`

Represents an interactive user interface element visible on screen.

### `class AppState(BaseModel)`

Represents the active application and on-screen context.

### `class UserContext(BaseModel)`

Root model for the complete user context state.
