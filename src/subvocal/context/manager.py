"""Context Manager for retrieving, filtering, and searching user context.

Integrates context querying with custom articulatory phonetic distance matching
to enable context-aware silent speech decoding.
"""

from typing import List, Tuple, Optional
from subvocal.shorthand.decoder import articulatory_distance
from subvocal.shorthand.spec import compress_word
from subvocal.context.schema import UserContext, Contact, UIElement, CalendarEvent


class ContextManager:
    """Manages active UserContext and provides phonetic searching utilities."""
    
    def __init__(self, context: UserContext):
        self.context = context
        
    def update_context(self, new_context: UserContext):
        """Update the active user context state."""
        self.context = new_context
        
    def get_contact_names(self) -> List[str]:
        """Get list of all contact names in plain text."""
        return [c.name for c in self.context.contacts]
        
    def get_contact_shorthands(self) -> List[str]:
        """Get list of all pre-computed contact name shorthands."""
        return [c.shorthand_name for c in self.context.contacts]
        
    def get_calendar_titles(self) -> List[str]:
        """Get list of all calendar event titles in plain text."""
        return [e.title for e in self.context.calendar]
        
    def get_calendar_shorthands(self) -> List[str]:
        """Get list of all calendar event title shorthands."""
        return [e.shorthand_title for e in self.context.calendar]
        
    def get_visible_element_labels(self) -> List[str]:
        """Get list of all visible UI element labels in plain text."""
        return [el.label for el in self.context.app_state.visible_elements]
        
    def get_visible_element_shorthands(self) -> List[str]:
        """Get list of all visible UI element shorthands."""
        return [el.shorthand_label for el in self.context.app_state.visible_elements]
        
    def get_all_context_words(self) -> List[str]:
        """Compile a list of all potential word tokens present in the active context.
        
        Useful for seeding the dictionary candidate generator in the decoder.
        """
        words = []
        
        # Add contact names
        for c in self.context.contacts:
            words.extend(c.name.split())
            
        # Add calendar event titles
        for e in self.context.calendar:
            words.extend(e.title.split())
            
        # Add visible UI elements labels
        for el in self.context.app_state.visible_elements:
            words.extend(el.label.split())
            
        # Add page title/app name
        words.extend(self.context.app_state.current_app.split())
        if self.context.app_state.page_title:
            words.extend(self.context.app_state.page_title.split())
            
        # De-duplicate and filter out empty words / punctuation
        clean_words = []
        seen = set()
        for w in words:
            clean_w = "".join(char for char in w if char.isalnum()).lower()
            if clean_w and clean_w not in seen:
                seen.add(clean_w)
                clean_words.append(w)  # Preserve original casing
                
        return clean_words

    def search_contacts(self, noisy_shorthand: str) -> List[Tuple[Contact, float]]:
        """Search contacts by comparing noisy shorthand against pre-computed contact shorthands.
        
        Returns:
            List of tuples (Contact, articulatory_distance) sorted by distance.
        """
        results = []
        tokens = noisy_shorthand.strip().split()
        
        for contact in self.context.contacts:
            # Match tokens of the contact name
            contact_tokens = contact.shorthand_name.split()
            
            # Simple alignment distance: sum of minimum alignment distances per token
            # For each token in the query, find best match in the contact name
            total_dist = 0.0
            for q_tok in tokens:
                best_tok_dist = 999.0
                for c_tok in contact_tokens:
                    dist = articulatory_distance(q_tok, c_tok)
                    if dist < best_tok_dist:
                        best_tok_dist = dist
                total_dist += best_tok_dist
                
            results.append((contact, total_dist))
            
        results.sort(key=lambda x: x[1])
        return results

    def search_elements(self, noisy_shorthand: str) -> List[Tuple[UIElement, float]]:
        """Search visible UI elements using articulatory distance matching.
        
        Returns:
            List of tuples (UIElement, articulatory_distance) sorted by distance.
        """
        results = []
        tokens = noisy_shorthand.strip().split()
        
        for el in self.context.app_state.visible_elements:
            el_tokens = el.shorthand_label.split()
            
            total_dist = 0.0
            for q_tok in tokens:
                best_tok_dist = 999.0
                for el_tok in el_tokens:
                    dist = articulatory_distance(q_tok, el_tok)
                    if dist < best_tok_dist:
                        best_tok_dist = dist
                total_dist += best_tok_dist
                
            results.append((el, total_dist))
            
        results.sort(key=lambda x: x[1])
        return results
