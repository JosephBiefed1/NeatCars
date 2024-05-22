import math
import random
import sys
import os

import neat
import neat.config
import pygame


WIDTH = 1920
HEIGHT = 1080

car_width = 60
car_height = 60

BORDER_COLOR = (255, 255, 255, 255)

current_generation = 0

class Car:
    def __init__(self):
        self.sprite = pygame.image.load("car.png")
        #self.sprite = pygame.image.load("car.png").convert()
        self.sprite = pygame.transform.scale(self.sprite, (car_width, car_height))


        self.position = [941, 853]
        self.angle = 0
        self.speed = 0
        self.speed_set = False
        self.centre = [self.position[0]+car_width/2, self.position[1]+car_height/2]
        self.radars = [] #list of radars
        self.drawing_radars = []

        self.alive = True
        self.distance = 0
        self.time = 0
        self.rotated_sprite = self.sprite
    
    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)

        self.drawing_radars(screen)

    def drawing_radars(self, screen):

        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.centre, position, 1)

            pygame.draw.circle(screen, (255, 0, 0), self.position, 5, 1)

    

    def check_collision(self, car_track):
        self.alive = True
        for point in self.corners:
            if car_track.get_at((int(point[0]), int(point[1]))) == (255, 255, 255, 255):
                self.alive = False
                break
    def check_radar(self, degree, car_track):
        length = 0
        x = self.centre[0] + math.cos(math.radians(self.angle + degree)) * length
        y = self.centre[1] + math.sin(math.radians(self.angle + degree)) * length
        while not car_track.get_at((int(x), int(y))) == (255, 255, 255, 255) and length < 300:
            length += 1
            x = self.centre[0] + math.cos(math.radians(self.angle + degree)) * length
            y = self.centre[1] + math.sin(math.radians(self.angle + degree)) * length
            

        distance = int(math.sqrt(math.pow(x - self.centre[0], 2) + math.pow(y - self.centre[1], 2)))
        self.radars.append([(x, y), distance])

    
    def update(self, car_track):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_centre(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed

        # prevent the car from going to close to the border
        self.position[0] = max(20, self.position[0])
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1
        #update for u position

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(20, self.position[1])
        self.position[1] = min(self.position[1], WIDTH - 120)

        self.centre = [int(self.position[0]+car_width/2), int(self.position[1]+car_height/2)]

        length = car_width /2

        #calculate the 4 corners
        left_top = [self.centre[0] + math.cos(math.radians(360 - (self.angle + 30)))*length, self.centre[1] + math.sin(math.radians(360 - (self.angle + 30)))*length]
        right_top = [self.centre[0] + math.cos(math.radians(360 - (self.angle + 150)))*length, self.centre[1] + math.sin(math.radians(360 - (self.angle + 150)))*length]
        left_bottom = [self.centre[0] + math.cos(math.radians(360 - (self.angle + 210)))*length, self.centre[1] + math.sin(math.radians(360 - (self.angle + 210)))*length]
        right_bottom = [self.centre[0] + math.cos(math.radians(360 - (self.angle + 330)))*length, self.centre[1] + math.sin(math.radians(360 - (self.angle + 330)))*length]

        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(car_track)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, car_track)

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]

        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)
        return return_values
    
    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (car_width /2)
    

    def rotate_centre(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
    
def run_simulation(genomes, config):
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load("map.png").convert()

    global current_generation

    current_generation += 1

    counter = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, c in enumerate(cars):
            output = nets[i].activate(c.get_data())
            choice = output.index(max(output))

            if choice == 0:
                c.angle += 5
            elif choice == 1:
                c.angle -= 5
            elif choice == 2:
                if (c.speed - 2 >=12):
                    c.speed -= 2
            else:
                c.speed += 2
        still_alive = 0
        for i, c in enumerate(cars):
            if c.is_alive():
                still_alive+=1
                c.update(game_map)
                genomes[i][1].fitness += c.get_reward()
        if still_alive == 0:
            break
        counter += 1
        if counter == 1000:
            break
        
        screen.blit(game_map, (0, 0))

        for car in cars:
            if car.is_alive():
                car.draw(screen)


        text = generation_font.render("Generation: " + str(current_generation), True, (0,0,0))
        text_rect = text.get_rect()
        text_rect.center = (863, 958)
        screen.blit(text, text_rect)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    config_path = "config.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(run_simulation, 100)