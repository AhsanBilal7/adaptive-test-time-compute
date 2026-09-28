"""Heuristic scripted policy for the Crafter environment."""

import re


class CrafterPolicy:
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def get_action(self, obs, plan=None):
        """
        Get action based on Crafter-specific rules.
        
        Args:
            obs: Current observation
            plan: Current plan (optional)
            
        Returns:
            tuple: (action, rule_name)
        """
        obs_text = str(obs).lower()
        
        for rule_name, rule_func in self.rules:
            action = rule_func(obs_text, plan)
            if action:
                return action, rule_name
        
        return "noop", "default"
    
    def _initialize_rules(self):
        """
        Initialize Crafter-specific rules.
        
        Returns:
            list: List of (rule_name, rule_function) tuples
        """
        return [
            ("health_critical", self._rule_health_critical),
            ("hunger_critical", self._rule_hunger_critical),
            ("thirst_critical", self._rule_thirst_critical),
            ("collect_wood", self._rule_collect_wood),
            ("collect_stone", self._rule_collect_stone),
            ("collect_food", self._rule_collect_food),
            ("collect_water", self._rule_collect_water),
            ("craft_tools", self._rule_craft_tools),
            ("build_shelter", self._rule_build_shelter),
            ("combat", self._rule_combat),
            ("explore", self._rule_explore),
        ]
    
    def _rule_health_critical(self, obs_text, plan):
        """
        Rule: If health is critical, prioritize safety.
        """
        if any(word in obs_text for word in ['health low', 'health: 1', 'health: 2', 'dying']):
            if 'plant' in obs_text or 'sapling' in obs_text:
                return "collect plant"
            if 'safe' in obs_text or 'shelter' in obs_text:
                return "move to shelter"
            return "do"
        return None
    
    def _rule_hunger_critical(self, obs_text, plan):
        """
        Rule: If hunger is critical, find food.
        """
        if any(word in obs_text for word in ['hunger: 0', 'hunger: 1', 'starving', 'hungry']):
            if 'cow' in obs_text or 'pig' in obs_text or 'chicken' in obs_text:
                return "attack"
            if 'apple' in obs_text or 'plant' in obs_text:
                return "collect plant"
            return "search food"
        return None
    
    def _rule_thirst_critical(self, obs_text, plan):
        """
        Rule: If thirst is critical, find water.
        """
        if any(word in obs_text for word in ['thirst: 0', 'thirst: 1', 'dehydrated']):
            if 'water' in obs_text or 'lake' in obs_text or 'river' in obs_text:
                return "drink"
            return "search water"
        return None
    
    def _rule_collect_wood(self, obs_text, plan):
        """
        Rule: Collect wood when trees are nearby and wood is needed.
        """
        if 'tree' in obs_text:
            if plan and any(word in str(plan).lower() for word in ['wood', 'craft', 'build']):
                return "do"
            if 'inventory' in obs_text and 'wood: 0' in obs_text:
                return "do"
        return None
    
    def _rule_collect_stone(self, obs_text, plan):
        """
        Rule: Collect stone when rocks are nearby and stone is needed.
        """
        if 'stone' in obs_text or 'rock' in obs_text:
            if plan and 'stone' in str(plan).lower():
                return "do"
            if 'inventory' in obs_text and 'stone: 0' in obs_text:
                return "do"
        return None
    
    def _rule_collect_food(self, obs_text, plan):
        """
        Rule: Collect food sources when available.
        """
        food_sources = ['plant', 'sapling', 'apple', 'cow', 'pig', 'chicken']
        if any(source in obs_text for source in food_sources):
            if 'hunger' in obs_text:
                hunger_match = re.search(r'hunger[:\s]+(\d+)', obs_text)
                if hunger_match and int(hunger_match.group(1)) < 5:
                    if any(animal in obs_text for animal in ['cow', 'pig', 'chicken']):
                        return "attack"
                    return "do"
        return None
    
    def _rule_collect_water(self, obs_text, plan):
        """
        Rule: Drink water when thirsty and water is available.
        """
        if any(word in obs_text for word in ['water', 'lake', 'river']):
            if 'thirst' in obs_text:
                thirst_match = re.search(r'thirst[:\s]+(\d+)', obs_text)
                if thirst_match and int(thirst_match.group(1)) < 5:
                    return "drink"
        return None
    
    def _rule_craft_tools(self, obs_text, plan):
        """
        Rule: Craft tools when materials are available.
        """
        if plan and any(word in str(plan).lower() for word in ['craft', 'tool', 'pickaxe', 'sword']):
            if 'wood' in obs_text and 'stone' in obs_text:
                if 'wood:' in obs_text or 'stone:' in obs_text:
                    return "craft"
        return None
    
    def _rule_build_shelter(self, obs_text, plan):
        """
        Rule: Build shelter when planned and materials available.
        """
        if plan and any(word in str(plan).lower() for word in ['build', 'shelter', 'house']):
            if 'wood' in obs_text:
                wood_match = re.search(r'wood[:\s]+(\d+)', obs_text)
                if wood_match and int(wood_match.group(1)) >= 4:
                    return "place table"
        return None
    
    def _rule_combat(self, obs_text, plan):
        """
        Rule: Combat when threatened by hostile mobs.
        """
        hostile_mobs = ['zombie', 'skeleton', 'spider']
        if any(mob in obs_text for mob in hostile_mobs):
            if 'health' in obs_text:
                health_match = re.search(r'health[:\s]+(\d+)', obs_text)
                if health_match and int(health_match.group(1)) > 3:
                    return "attack"
            return "move away"
        return None
    
    def _rule_explore(self, obs_text, plan):
        """
        Rule: Explore when no immediate needs.
        """
        if plan and 'explore' in str(plan).lower():
            directions = ['north', 'south', 'east', 'west']
            for direction in directions:
                if direction not in obs_text:
                    return f"move {direction}"
        return None