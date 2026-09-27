#!/usr/bin/env ruby
# frozen_string_literal: true

require "date"
require "yaml"

errors = []

def load_yaml(path)
  YAML.safe_load_file(path, permitted_classes: [Date, Time], aliases: true) || []
end

def required_string!(errors, path, label, object, key)
  value = object[key]
  return if value.is_a?(String) && !value.strip.empty?

  errors << "#{path}: #{label} is missing #{key.inspect}"
end

projects = load_yaml("_data/projects.yml")
projects.each_with_index do |project, index|
  label = "project ##{index + 1}"
  required_string!(errors, "_data/projects.yml", label, project, "name")
  required_string!(errors, "_data/projects.yml", label, project, "blurb")

  stack = project["stack"]
  errors << "_data/projects.yml: #{label} stack must be a non-empty array" unless stack.is_a?(Array) && stack.any?

  url = project["url"]
  repo = project["repo"]
  links = project["links"]
  visibility = project["visibility"]
  has_link = [url, repo].any? { |value| value.is_a?(String) && !value.empty? && value != "#" } ||
    (links.is_a?(Array) && links.any?)

  errors << "_data/projects.yml: #{label} needs url, repo, links, or visibility" unless has_link || visibility
end

publications = load_yaml("_data/publications.yml")
publications.each_with_index do |publication, index|
  label = "publication ##{index + 1}"
  %w[title authors venue].each do |key|
    required_string!(errors, "_data/publications.yml", label, publication, key)
  end
  errors << "_data/publications.yml: #{label} is missing a numeric year" unless publication["year"].is_a?(Integer)

  links = publication["links"]
  next unless links

  unless links.is_a?(Array)
    errors << "_data/publications.yml: #{label} links must be an array"
    next
  end

  links.each_with_index do |link, link_index|
    required_string!(errors, "_data/publications.yml", "#{label} link ##{link_index + 1}", link, "label")
    required_string!(errors, "_data/publications.yml", "#{label} link ##{link_index + 1}", link, "url")
  end
end

reading = load_yaml("_data/reading.yml")
reading.each_with_index do |item, index|
  label = "reading item ##{index + 1}"
  %w[title url kind category note].each do |key|
    required_string!(errors, "_data/reading.yml", label, item, key)
  end

  tags = item["tags"]
  errors << "_data/reading.yml: #{label} tags must be a non-empty array" unless tags.is_a?(Array) && tags.any?

  saved_on = item["saved_on"]
  next unless saved_on

  valid_saved_on = saved_on.is_a?(Date) || saved_on.is_a?(Time) ||
    (saved_on.is_a?(String) && saved_on.match?(/\A\d{4}-\d{2}-\d{2}\z/))
  errors << "_data/reading.yml: #{label} saved_on must be YYYY-MM-DD" unless valid_saved_on
end

Dir["_posts/*.md"].sort.each do |path|
  raw = File.read(path)
  match = raw.match(/\A---\s*\n(.*?)\n---\s*\n/m)
  unless match
    errors << "#{path}: missing front matter"
    next
  end

  data = YAML.safe_load(match[1], permitted_classes: [Date, Time], aliases: true) || {}
  required_string!(errors, path, "post", data, "title")
  required_string!(errors, path, "post", data, "lede")

  %w[categories tags].each do |key|
    value = data[key]
    errors << "#{path}: #{key} must be a non-empty array" unless value.is_a?(Array) && value.any?
  end
end

if errors.any?
  warn errors.join("\n")
  exit 1
end

puts "Content validation passed"
