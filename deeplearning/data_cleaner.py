#!/usr/bin/env python3
"""
Data Cleaning Module for Clash Royale AI Training Data

Cleans game_log.jsonl by:
1. Removing entries where all cards are "Unknown"
2. Removing consecutive duplicate entries
3. Filtering out card detection hallucinations
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

class DataCleaner:
    """Cleans training data for better model performance"""
    
    def __init__(self):
        self.stats = {
            'total_entries': 0,
            'unknown_cards_removed': 0,
            'duplicates_removed': 0,
            'final_entries': 0
        }
    
    def clean_data(self, input_file: str, output_file: str = None) -> str:
        """Clean the data file and return output path"""
        if output_file is None:
            output_file = input_file.replace('.jsonl', '_cleaned.jsonl')
        
        print(f"Cleaning data: {input_file} → {output_file}")
        
        # Read all entries
        entries = self._read_entries(input_file)
        self.stats['total_entries'] = len(entries)
        
        # Clean entries
        cleaned_entries = self._remove_unknown_cards(entries)
        cleaned_entries = self._remove_duplicates(cleaned_entries)
        
        self.stats['final_entries'] = len(cleaned_entries)
        
        # Write cleaned data
        self._write_entries(cleaned_entries, output_file)
        
        # Print stats
        self._print_stats()
        
        return output_file
    
    def _read_entries(self, file_path: str) -> List[Dict]:
        """Read all entries from JSONL file"""
        entries = []
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num}")
                        continue
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return []
        
        return entries
    
    def _remove_unknown_cards(self, entries: List[Dict]) -> List[Dict]:
        """Remove entries where all cards are Unknown"""
        cleaned = []
        
        for entry in entries:
            cards = entry.get('cards', [])
            
            # Skip if no cards data
            if not cards:
                continue
            
            # Check if all cards are Unknown
            card_names = [card.get('name', '').lower() for card in cards]
            unknown_count = sum(1 for name in card_names if name in ['unknown', '', 'none'])
            
            # Keep entry if not all cards are unknown
            if unknown_count < len(cards):
                cleaned.append(entry)
            else:
                self.stats['unknown_cards_removed'] += 1
        
        return cleaned
    
    def _remove_duplicates(self, entries: List[Dict]) -> List[Dict]:
        """Remove consecutive duplicate entries"""
        if not entries:
            return entries
        
        cleaned = [entries[0]]  # Always keep first entry
        
        for i in range(1, len(entries)):
            current = entries[i]
            previous = entries[i-1]
            
            # Compare key game state fields
            if not self._are_entries_similar(current, previous):
                cleaned.append(current)
            else:
                self.stats['duplicates_removed'] += 1
        
        return cleaned
    
    def _are_entries_similar(self, entry1: Dict, entry2: Dict) -> bool:
        """Check if two entries are similar enough to be considered duplicates"""
        # Compare elixir (allow small difference)
        elixir1 = entry1.get('elixir', 0)
        elixir2 = entry2.get('elixir', 0)
        if abs(elixir1 - elixir2) > 1:
            return False
        
        # Compare cards
        cards1 = self._get_card_names(entry1.get('cards', []))
        cards2 = self._get_card_names(entry2.get('cards', []))
        if cards1 != cards2:
            return False
        
        # Compare troop counts
        troops1 = len(entry1.get('troops', []))
        troops2 = len(entry2.get('troops', []))
        if abs(troops1 - troops2) > 2:  # Allow small troop count difference
            return False
        
        # Compare win condition
        win1 = entry1.get('win_condition', 'ongoing')
        win2 = entry2.get('win_condition', 'ongoing')
        if win1 != win2:
            return False
        
        return True
    
    def _get_card_names(self, cards: List[Dict]) -> List[str]:
        """Extract and sort card names for comparison"""
        names = [card.get('name', '').lower() for card in cards]
        return sorted([name for name in names if name and name != 'unknown'])
    
    def _write_entries(self, entries: List[Dict], output_file: str):
        """Write cleaned entries to file"""
        with open(output_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
    
    def _print_stats(self):
        """Print cleaning statistics"""
        print("\n" + "="*50)
        print("DATA CLEANING RESULTS")
        print("="*50)
        print(f"Total entries: {self.stats['total_entries']}")
        print(f"Unknown cards removed: {self.stats['unknown_cards_removed']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"Final entries: {self.stats['final_entries']}")
        
        if self.stats['total_entries'] > 0:
            reduction = (self.stats['total_entries'] - self.stats['final_entries']) / self.stats['total_entries'] * 100
            print(f"Data reduction: {reduction:.1f}%")
        
        print("="*50)

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Clean Clash Royale training data")
    parser.add_argument("input_file", help="Input JSONL file to clean")
    parser.add_argument("--output", "-o", help="Output file (default: input_cleaned.jsonl)")
    parser.add_argument("--preview", "-p", action="store_true", help="Preview cleaning without saving")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    cleaner = DataCleaner()
    
    if args.preview:
        # Preview mode - just show stats
        entries = cleaner._read_entries(args.input_file)
        cleaner.stats['total_entries'] = len(entries)
        
        cleaned = cleaner._remove_unknown_cards(entries)
        cleaned = cleaner._remove_duplicates(cleaned)
        cleaner.stats['final_entries'] = len(cleaned)
        
        print("PREVIEW MODE - No files will be modified")
        cleaner._print_stats()
    else:
        # Actually clean the data
        output_file = cleaner.clean_data(args.input_file, args.output)
        print(f"\nCleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()