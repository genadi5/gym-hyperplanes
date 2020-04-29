class MissedAreas:
    def __init__(self, missed_areas, hp_state, hp_dist):
        # missed areas is a map of area to part of dataset instances which falls into this area
        # on next iteration this will be our 'world'
        self.missed_areas = missed_areas
        self.hp_state = hp_state
        self.hp_dist = hp_dist

    def get_missed_areas(self):
        return self.missed_areas

    def get_hp_state(self):
        return self.hp_state

    def get_hp_dist(self):
        return self.hp_dist
