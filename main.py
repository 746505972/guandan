from Game import GuandanGame, GameConfig


def main():
    config: GameConfig = GameConfig()
    config.player_0 = 'Human'
    game: GuandanGame = GuandanGame(config)
    game.play_game()


if __name__ == "__main__":
    main()
